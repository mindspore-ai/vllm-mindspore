#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import copy
from typing import Dict, List, Optional, Set, Tuple, Iterable
import numpy as np
import re
import mindspore as ms
from mindspore import Parameter, Tensor, mint, nn, jit, ops, mutable
from mindspore.common import lazy_inline
from mindspore.communication import get_rank
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer, Normal
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.auto_generate import (MoeInitRoutingV2, MoeTokenUnpermute)

import vllm.envs as envs
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.attention.backends.abstract import AttentionType
from vllm.forward_context import get_forward_context

from vllm_mindspore.attention import Attention
from vllm_mindspore.model_executor.layers.layernorm import RMSNorm
from vllm_mindspore.model_executor.layers.linear import (QKVParallelLinear, RowParallelLinear)
from vllm_mindspore.model_executor.layers.moe import (SharedParallelMLP, RoutedParallelMLP)
from vllm_mindspore.model_executor.layers.logits_processor import LogitsProcessor
from vllm_mindspore.model_executor.layers.rotary_embedding import get_rope
from vllm_mindspore.model_executor.layers.sampler import (SamplerOutput, get_sampler)
from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import (ParallelLMHead, VocabParallelEmbedding)
from vllm_mindspore.model_executor.model_loader.weight_utils import default_weight_loader
from vllm_mindspore.model_executor.models.utils import maybe_prefix
from vllm_mindspore.model_executor.sampling_metadata import SamplingMetadata
from vllm_mindspore.model_executor.models.model_base import MsModelBase, Fake_Attention, Fake_Attention_V1
from vllm_mindspore.model_executor.models.attention_mask import LowerTriangularMask, PFALowerTriangularMask
from vllm_mindspore.utils import STR_DTYPE_TO_MS_DTYPE, atlas_inference
from vllm_mindspore.distributed.communication_op import ReduceFromModelParallelRegion
from vllm_mindspore.v1.attention.backends.flash_attn import FlashAttentionMetadata


def _pad_inputs_using_pfa_mode_length(origin_inputs, pad_token_id):
    pad_length = (origin_inputs.shape[1] + 127) // 128 * 128 - origin_inputs.shape[
        -1
    ]
    input_ids = ops.pad(
        origin_inputs,
        (0, pad_length),
        "constant",
        pad_token_id,
    )
    return input_ids


def _pad_slot_mapping_using_pfa_mode_length(origin_slot_mapping, batch_size):
    slot_mapping = origin_slot_mapping.reshape(batch_size, -1)
    pad_length = (slot_mapping.shape[-1] + 127) // 128 * 128 - slot_mapping.shape[
        -1
    ]
    slot_mapping = ops.pad(
        slot_mapping,
        (0, pad_length),
        "constant",
        -1,
    )
    slot_mapping = slot_mapping.reshape(-1)
    return slot_mapping


def _pad_inputs(inputs, batch_valid_length, pad_value=0):
    input_list = []
    offset = 0
    max_seq_len = max(batch_valid_length)
    for seq_len in batch_valid_length:
        input_list.append(
            ops.pad(inputs[offset : offset + seq_len], (0, max_seq_len - seq_len), "constant", pad_value)
        )
        offset += seq_len
    inputs = ops.cat(input_list)
    return inputs

def _unpadding_outputs(hidden_states, batch_valid_length):
    hidden_states_list = []
    for idx, seq_len in enumerate(batch_valid_length):
        hidden_states_list.append(hidden_states[idx, : int(seq_len)])
    hidden_states = ops.cat(hidden_states_list)
    return hidden_states

class Linear(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 init_method_std=0.01,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None,
                 transpose_b=True,
                 expert_num=1,
                 outer_batch=1,
                 expert_group_size=None,
                 use_gmm=False,
                 param_init_type=mstype.float32,
                 compute_dtype=mstype.float16):
        super(Linear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if not (isinstance(activation, str) or activation is None or issubclass(activation, nn.Cell)):
            raise TypeError(f"For Linear cell, the activation should str type or nn.Cell type, but got {activation}.")

        transpose_b = False if use_gmm else transpose_b
        if weight_init == "normal":
            weight_init = Normal(sigma=init_method_std, mean=0)
        weight_shape = [out_channels, in_channels] if transpose_b else [in_channels, out_channels]
        if isinstance(weight_init, Tensor) and (weight_init.ndim != 2 or weight_init.shape[0] != weight_shape[0] or
                                                weight_init.shape[1] != weight_shape[1]):
            raise ValueError("The shape of parameter 'weight_init' is error, please check shape of 'weight_init'.")

        self.expert_num = expert_num
        self.outer_batch = outer_batch
        self.expert_group_size = expert_group_size
        self.use_gmm = use_gmm
        self.transpose_b = transpose_b

        self.weight = Parameter(initializer(weight_init, weight_shape, param_init_type), name="weight")
        self.matmul = P.MatMul(transpose_b=transpose_b)

        self.bias = None
        self.has_bias = has_bias
        if self.has_bias:
            if isinstance(bias_init, Tensor) and (bias_init.ndim != 1 or bias_init.shape[0] != out_channels):
                raise ValueError("The shape of parameter 'bias_init' is error, please check shape of 'bias_init'.")

            self.bias = Parameter(initializer(bias_init, [out_channels], param_init_type), name="bias")
            self.bias.parallel_optimizer = False
            self.bias_add = P.Add()

        self.param_init_type = param_init_type
        self.dtype = compute_dtype
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.shape = P.Shape()

    def construct(self, x, group_list=None):
        """Forward process, x should be a tensor"""
        out_shape = self.shape(x)[:-1] + (self.out_channels,)
        x = self.reshape(x, (-1, self.in_channels))
        ori_dtype = F.dtype(x)
        weight = self.cast(self.weight, self.dtype)
        x = self.cast(x, self.dtype)
        x = self.matmul(x, weight)
        if self.has_bias:
            x = self.bias_add(x, self.cast(self.bias, self.dtype))
        x = F.cast(x, ori_dtype)
        output = self.reshape(x, out_shape)
        return output

class MoeGatingGroupTopkCell(nn.Cell):
    def __init__(self):
        super().__init__()
        self.moe_gating_group_topk = ms.ops.MoeGatingGroupTopK()

    def construct(self, x, bias=None, k=8, k_group=8, group_count=8, group_select_mode=0,
                  renorm=0, norm_type=0, out_flag=False, routed_scaling_factor=1.0, eps=1e-20):
        x = x.reshape((-1, x.shape[-1]))
        y, expert_idx, _ = self.moe_gating_group_topk(
            x, bias, k, k_group, group_count, group_select_mode, renorm, norm_type, out_flag, routed_scaling_factor, eps)
        return y, expert_idx


class PanguMoE(nn.Cell):
    """
    expert layer.

    Args:
        config: configuration
        submodules: reserve arguments, not used now.
        layer_number: reserve arguments, not used now.
    """
    # pylint: disable=C0103
    def __init__(self, config, parallel_config, additional_config, quant_config: Optional[QuantizationConfig] = None, prefix: str = ""):
        super().__init__()

        self.mstype = STR_DTYPE_TO_MS_DTYPE.get(config.torch_dtype)
        self.tp_group_size = parallel_config.tensor_parallel_size
        self.ep_group_size = parallel_config.tensor_parallel_size

        rank_id = int(get_rank())
        self.rank = get_rank()

        self.num_local_experts = config.num_experts // self.ep_group_size
        local_expert_indices = [rank_id * self.num_local_experts + i for i in range(self.num_local_experts)]

        for x in local_expert_indices:
            if x >= config.num_experts:
                raise ValueError(f"expect all local expert indices < expert num, but got {local_expert_indices}")

        self.experts = RoutedParallelMLP(config, parallel_config,prefix=f'{prefix}.experts',
                                         quant_config=quant_config)

        if config.shared_expert_intermediate_size:
            self.shared_expert = SharedParallelMLP(config, parallel_config,
                                                   prefix=f'{prefix}.shared_expert',
                                                   quant_config=quant_config)

            self.router_scale = Parameter((mint.ones((1, config.num_experts), dtype=self.mstype)))
            self.use_shared_expert = True

        self.reduce_from_mp_region = ReduceFromModelParallelRegion()
        self.quant_config = quant_config
        self.expert_group = int(get_rank())
        self.router_topk = config.num_experts_per_tok
        self.index = Tensor(ops.range(0, self.num_local_experts, 1), ms.int64)
        self.group_num = self.router_topk // self.ep_group_size
        self.index += self.expert_group * self.num_local_experts
        if self.group_num == 1:
            self.expert_group_tensor = Tensor([self.expert_group,], ms.int32)
        else:
            begin = self.group_num * self.expert_group
            end = self.group_num * (self.expert_group + 1)
            self.expert_group_tensor = Tensor(ops.range(begin, end, 1), ms.int32)

        self.use_ep = True if self.ep_group_size > 1 else False
        self.moe_init_routing_v2 = MoeInitRoutingV2()
        self.moe_token_unpermute = MoeTokenUnpermute()
        self.moe_gating_topk_op = MoeGatingGroupTopkCell()

        self.gate = Linear(
            config.hidden_size,
            config.num_experts,
            has_bias=False,
            param_init_type=self.mstype,
            compute_dtype=self.mstype
        )
        self.bias = mint.arange(
            0,
            config.num_experts,
            config.num_experts_per_tok,
            dtype=ms.int64).unsqueeze(0)

        self.atlas_inference = atlas_inference()
        self.enable_high_precision = additional_config is not None and additional_config.get('enable_high_precision', False)
        self.enable_moe_fusion_kernel = not (self.atlas_inference and self.enable_high_precision)
        self.moe_fusion_support_dtype =  ms.float16 if self.atlas_inference else ms.bfloat16

    def token_permutation(self, hidden_states, global_indices):
        hidden_shape = hidden_states.shape
        # [S/TP, B, H] -> [S*B/TP, H]
        hidden_states = hidden_states.view(-1, hidden_shape[-1])

        local_indices = mint.index_select(
            global_indices, 1, self.expert_group_tensor
        ).view(-1)

        expert_mask = (
            local_indices.to(ms.int64).view(-1, 1)
            == self.index.to(ms.int64).view(1, self.num_local_experts)
        ).to(ms.int32)

        tokens_per_expert = mint.sum(expert_mask, dim=0)
        local_indices = local_indices.to(ms.float32)
        _, indices = mint.topk(
            local_indices, local_indices.shape[0], largest=False, sorted=True
        )

        indices_ = mint.div(indices, self.group_num, rounding_mode="floor")
        permuted_local_hidden_states = mint.index_select(
            hidden_states, 0, indices_
        )

        return (permuted_local_hidden_states, indices, tokens_per_expert)


    def token_unpermutation(self, permuted_tokens, sorted_indices, probs, hidden_shape):
        scores = probs.to(dtype=permuted_tokens.dtype)
        unpermuted_local_hidden = mint.index_select(
            permuted_tokens, 0, sorted_indices
        )

        # Scale the expert output prior to reduction and subsequent to local unpermutation if k > 1.
        if self.router_topk > 1:
            unpermuted_local_hidden = unpermuted_local_hidden * scores.view(-1, 1)

        output_total = unpermuted_local_hidden

        # Unpermute the tokens across expert parallel devices.
        if self.use_ep:
            unpermuted_global_hidden = unpermuted_local_hidden
            # move allreduce after expert + shared_expert
            if self.tp_group_size != self.ep_group_size:
                output_total = self.all_reduce(unpermuted_global_hidden)
            else:
                output_total = unpermuted_global_hidden

        if self.router_topk == 1:
            output_total = output_total * scores
        output_total = output_total.view(hidden_shape)

        if self.group_num > 1:
            output_total = output_total.reshape(-1, self.group_num, hidden_shape[-1])
            output_total = mint.sum(output_total, dim=1)
        return output_total

    def construct(self, hidden_states: ms.Tensor, is_prefill: bool):
        """moe layer forward"""
        hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_shape[-1])

        # gating and topk
        logits = self.gate(hidden_states)
        if not self.atlas_inference:
            scores, indices = self.moe_gating_topk_op(logits)
        else:
            logits = self.cast(logits, mstype.float32)
            scores = mint.nn.functional.softmax(logits, dim=-1, dtype=mstype.float32)
            scores = self.cast(scores, mstype.float16)
            topk_weight, topk_idx = mint.max(scores.view(scores.shape[0], 8, -1), dim=-1)
            topk_idx = topk_idx + self.bias
            scores, indices = topk_weight, self.cast(topk_idx, ms.int64)

        # initrouting
        indices = self.cast(indices, ms.int64)

        if self.atlas_inference and (self.enable_high_precision or is_prefill):
            dispatched_input, dispatched_indices, group_list = self.token_permutation(
                hidden_states, indices
            )
            _, unsort_map = mint.topk(dispatched_indices.to(ms.float32), dispatched_indices.shape[0], dim=0,
                                            largest=False, sorted=True)
            unsort_map = unsort_map.to(ms.int32)
            group_list = mint.cumsum(group_list.to(ms.int32), dim=0)
            cumsum_flag = True
        else:
            local_indices = mint.index_select(indices, 1, self.expert_group_tensor).astype(ms.int32)
            local_indices = local_indices - (self.expert_group * self.num_local_experts)
            cumsum_flag = self.atlas_inference
            dispatched_input, unsort_map, group_list, _ = \
                self.moe_init_routing_v2(
                    hidden_states,
                    local_indices,
                    active_num=0,
                    expert_capacity=0,
                    expert_num=self.num_local_experts,
                    drop_pad_mode=0,
                    expert_tokens_count_or_cumsum_flag=1 if cumsum_flag else 2, # use cumsum
                    expert_tokens_before_capacity_flag=not cumsum_flag)
        # route experts
        group_list_i64 = self.cast(group_list, ms.int64)
        expert_output = self.experts(dispatched_input, group_list_i64, cumsum_flag)

        # unpermute token
        select_router_scale = mint.gather(self.router_scale.reshape(-1), 0, indices.reshape(-1))
        select_router_scale = select_router_scale.reshape(indices.shape)
        scores = scores * select_router_scale
        if self.router_topk > 1:
            local_probs = mint.index_select(scores, 1, self.expert_group_tensor)
        else:
            local_probs = scores
        if self.enable_moe_fusion_kernel:
            output = self.moe_token_unpermute(
                permuted_tokens=expert_output.astype(self.moe_fusion_support_dtype),
                sorted_indices=unsort_map,
                probs=local_probs.astype(self.moe_fusion_support_dtype),
                padded_mode=False,
                restore_shape=None,
            )
            output = self.cast(output, self.mstype)
        else:
            output = self.token_unpermutation(
                permuted_tokens=expert_output,
                sorted_indices=unsort_map,
                probs=local_probs,
                hidden_shape=hidden_shape,
            )
        output = output.reshape(hidden_shape)

        # shared experts
        if self.use_shared_expert:
            shared_expert_output = self.shared_expert(hidden_states)
            shared_expert_output = shared_expert_output.reshape(hidden_shape)
            output = output + shared_expert_output
            if self.ep_group_size == self.tp_group_size:
                output = self.reduce_from_mp_region(output)
        return output


class PanguAttention(nn.Cell):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            num_kv_heads: int,
            max_position: int = 4096 * 32,
            rope_theta: float = 10000,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
            rope_scaling: Optional[Tuple] = None,
            prefix: Optional[str] = "",
            attn_type: Optional[str] = AttentionType.DECODER,
            params_dtype: Optional[mstype.Type] = mstype.bfloat16,
            use_pfa: bool = False
    ) -> None:
        super().__init__()

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            params_dtype=params_dtype,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=True,
            skip_bias_add=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            params_dtype=params_dtype,
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
            dtype=params_dtype,
            use_pfa=use_pfa,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            use_pfa=use_pfa,
            params_dtype=params_dtype,
        )
        self.rank = get_rank()
        self.dump = ms.ops.TensorDump()

    def construct(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        is_prefill: bool,
        slot_mapping: Tensor,
        attn_mask: Tensor,
        batch_valid_length: Tensor,
        q_seq_lens: Tensor,
        block_tables: Tensor,
    ) -> Tensor:
        # hidden_states: prefill (1, t, h), decode (b, 1, h)
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = mint.split(qkv, (self.q_size, self.kv_size, self.kv_size), -1)
        q, k = self.rotary_emb(positions, q, k, batch_valid_length, is_prefill)

        # attn_output: prefill (1, t, h), decode (b, 1, h)
        attn_output = self.attn(q, k, v, key_cache, value_cache, is_prefill, slot_mapping, attn_mask,
                                batch_valid_length, q_seq_lens, block_tables)

        output, wo_bias = self.o_proj(attn_output)
        output = mint.add(output, wo_bias)
        return output


def lazy_inline_factory(func):
    def wrapper(*args, **kwargs):
        if not atlas_inference():
            return lazy_inline(func)(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper


class PanguDecodeLayer(nn.Cell):
    @lazy_inline_factory
    def __init__(
        self,
        config,
        parallel_config,
        additional_config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config)

        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        self.mstype = STR_DTYPE_TO_MS_DTYPE.get(config.torch_dtype) # ms.bfloat16/float16
        self.use_pfa = additional_config is not None and additional_config.get('enable_high_precision', False)
        self.self_attn = PanguAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=AttentionType.DECODER,
            params_dtype=self.mstype,
            use_pfa=self.use_pfa
        )

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            params_dtype=self.mstype,
        )

        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            params_dtype=self.mstype,
        )

        self.mlp = PanguMoE(config, parallel_config, additional_config, prefix=f"{prefix}.mlp", quant_config=quant_config)
        self.no_inline = False

    def construct(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        is_prefill: bool,
        slot_mapping: Tensor,
        attn_mask: Tensor,
        batch_valid_length: Tensor,
        q_seq_lens: Tensor,
        block_tables: Tensor,
    ):
        """Construct function of transformer layer."""
        norm_output = self.input_layernorm(hidden_states)
        attention_output = self.self_attn(
            positions,
            norm_output,
            key_cache,
            value_cache,
            is_prefill,
            slot_mapping,
            attn_mask,
            batch_valid_length,
            q_seq_lens,
            block_tables
        )

        norm_input = ops.add(hidden_states, attention_output)
        norm_output = self.post_attention_layernorm(norm_input)
        mlp_output = self.mlp(norm_output, is_prefill)
        output = ops.add(norm_input, mlp_output)
        return output


class Pangu72BModel(nn.Cell):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config
        additional_config = vllm_config.additional_config
        self.quant_config = vllm_config.quant_config
        self.mstype = STR_DTYPE_TO_MS_DTYPE.get(config.torch_dtype)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            params_dtype=STR_DTYPE_TO_MS_DTYPE.get(config.torch_dtype),
            padding_size=1,
            quant_config=self.quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.CellList()
        for layer_id in range(config.num_hidden_layers):
            layer = PanguDecodeLayer(config=config, parallel_config=parallel_config, additional_config=additional_config,
                                     cache_config=vllm_config.cache_config, quant_config=self.quant_config,
                                     prefix=f'{prefix}.layers.{layer_id}')
            self.layers.append(layer)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps,
                            params_dtype=STR_DTYPE_TO_MS_DTYPE.get(config.torch_dtype))
        self.num_layers = config.num_hidden_layers
        self.rank = int(get_rank())
        self.expert_num = config.num_experts

    @jit
    def construct(
        self,
        input_ids: Optional[Tensor],
        positions: Tensor,
        key_caches: List[Tensor],
        value_caches: List[Tensor],
        is_prefill: bool,
        slot_mapping: Tensor,
        attn_mask: Tensor,
        batch_valid_length: Tensor,
        q_seq_lens: Tensor,
        block_tables: Tensor
    ):
        # tokens: [bs, seq/1]
        hidden_states = self.cast(self.embed_tokens(input_ids), self.mstype)
        # h: [bs, seq/1, hidden_dim]
        for i in range(self.num_layers):
            hidden_states = self.layers[i](
                positions,
                hidden_states,
                key_caches[i],
                value_caches[i],
                is_prefill,
                slot_mapping,
                attn_mask,
                batch_valid_length,
                q_seq_lens,
                block_tables,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def find_gate_up_expert(self, s):
        pattern = r"^model\.\d+\.mlp\.experts\.\d+\.gate_up_proj\..*$"
        match = re.match(pattern, s)
        if match:
            second_number_match = re.search(r"\.\d+(?=\.gate_up_proj\.)", s)
            if second_number_match:
                expert_idx = int(second_number_match.group(0).strip('.'))
                new_name = re.sub(r"\.\d+(?=\.gate_up_proj\.)", "", s, count=1)
                return new_name, expert_idx
        return s, None

    def find_down_expert(self, s):
        pattern = r"^model\.\d+\.mlp\.experts\.\d+\.down_proj\..*$"
        match = re.match(pattern, s)
        if match:
            second_number_match = re.search(r"\.\d+(?=\.down_proj\.)", s)
            if second_number_match:
                expert_idx = int(second_number_match.group(0).strip('.'))
                new_name = re.sub(r"\.\d+(?=\.down_proj\.)", "", s, count=1)
                return new_name, expert_idx
        return s, None

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]], params_dict: Dict[str, Parameter]):
        loaded_params: Set[str] = set()
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        dynamic_quant = hasattr(self.quant_config, "dynamic_quant") and self.quant_config.dynamic_quant == True
        need_find_expert_idx = self.quant_config is None or dynamic_quant == False
        for name, loaded_weight in weights:
            name = name.replace(".layers", "")
            name = name.replace("k_proj.kv_cache_scale", "attn.k_scale")
            name = name.replace("v_proj.kv_cache_scale", "attn.v_scale")
            name = name.replace("k_proj.kv_cache_offset", "attn.k_offset")
            name = name.replace("v_proj.kv_cache_offset", "attn.v_offset")
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if "experts" in name and need_find_expert_idx:
                    name, expert_idx  = self.find_gate_up_expert(name)
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    if "experts" in name and need_find_expert_idx:
                        weight_loader(param, loaded_weight, expert_idx, shard_id)
                    else:
                        weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(name)
                    break
            else:
                if "experts" in name and need_find_expert_idx:
                    name, expert_idx = self.find_down_expert(name)
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    if "experts" in name and need_find_expert_idx:
                        weight_loader(param, loaded_weight, expert_idx)
                    else:
                        weight_loader(param, loaded_weight)
                    loaded_params.add(name)
        
        def adjust_weight(params_dict):
            if not atlas_inference():
                return

            target_keywords = [
                "qkv_proj.weight",
                "o_proj.weight",
                "experts.gate_up_proj.weight",
                "experts.down_proj.weight",
                "shared_expert.gate_up_proj.weight",
                "shared_expert.down_proj.weight",
                "gate.weight",
            ]

            for name, param in params_dict.items():
                if any(name.endswith(keyword) for keyword in target_keywords):
                    cast_weight = ops.auto_generate.format_cast(param, 29)
                    ms.runtime.synchronize()
                    param.set_data(cast_weight)

        ms.runtime.synchronize()
        adjust_weight(params_dict)
        ms.runtime.synchronize()

        return loaded_params


class PanguProMoEForCausalLM(MsModelBase):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.mstype = STR_DTYPE_TO_MS_DTYPE.get(self.config.torch_dtype)
        self.enable_high_precision = self.additional_config is not None and \
                                     self.additional_config.get('enable_high_precision', False)
        self.quant_config = vllm_config.quant_config
        self.model = Pangu72BModel(vllm_config=vllm_config,
                                   prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(self.config.vocab_size,
                                      self.config.hidden_size,
                                      params_dtype=self.mstype,
                                      padding_size=1,
                                      prefix=maybe_prefix(prefix, "lm_head"))
        self.set_modules({"model": self.model, "lm_head": self.lm_head})
        self.prefill = True
        self.set_flags = False

        if not self.enable_high_precision:
            self.casual_mask = LowerTriangularMask(dtype=self.mstype,
                                                   max_model_len=self.model_config.max_model_len)
        else:
            self.casual_mask = PFALowerTriangularMask(dtype=self.mstype,
                                                      max_model_len=self.model_config.max_model_len)
        kv_cache_dtype = self.model_config.dtype if self.cache_config.cache_dtype == "auto" \
            else self.cache_config.cache_dtype
        if kv_cache_dtype in STR_DTYPE_TO_MS_DTYPE:
            kv_cache_dtype = STR_DTYPE_TO_MS_DTYPE[kv_cache_dtype]
        self.kv_cache_dtype = kv_cache_dtype
        self.set_model_inputs(self.prefill)

        if envs.VLLM_USE_V1:
            self.kv_caches = [Fake_Attention_V1(self.kv_cache_dtype) for i in range(self.config.num_hidden_layers)]
        else:
            self.kv_caches = [Fake_Attention(self.kv_cache_dtype) for i in range(self.config.num_hidden_layers)]

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(self.config.num_hidden_layers):
            compilation_config.static_forward_context[str(i)] = self.kv_caches[i]
        
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.sampler = get_sampler()


    def set_model_inputs(self, is_prefill):
        dyn_input_ids = Tensor(shape=[None, None], dtype=mstype.int64)
        dyn_position_ids = Tensor(shape=[None], dtype=mstype.int64)

        block_size = self.cache_config.block_size
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()
        if atlas_inference():
            kv_cache_shape = (None, block_size, num_kv_heads * head_size)
        else:
            kv_cache_shape = (None, block_size, num_kv_heads, head_size)

        num_layers = self.model_config.get_num_layers(self.parallel_config)

        dyn_key_cache = Tensor(shape=kv_cache_shape, dtype=self.kv_cache_dtype)
        dyn_value_cache = Tensor(shape=kv_cache_shape, dtype=self.kv_cache_dtype)
        dyn_key_caches = mutable([dyn_key_cache for _ in range(num_layers)])
        dyn_value_caches = mutable([dyn_value_cache for _ in range(num_layers)])
        dyn_slot_mapping = Tensor(shape=[None, ], dtype=mstype.int32)
        if is_prefill and self.enable_high_precision and atlas_inference():
            dynamic_attention_mask = Tensor(
                shape=[None, None, None, None], dtype=mstype.bool_
            )
        else:
            dynamic_attention_mask = Tensor(shape=[None, None], dtype=self.mstype if \
                not self.enable_high_precision or atlas_inference() else mstype.bool_)

        dyn_batch_valid_length = Tensor(shape=[None,], dtype=mstype.int32)
        dyn_q_seq_lens = Tensor(shape=[None, ], dtype=mstype.int32) if not atlas_inference() else None
        dyn_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        self.model.set_inputs(
            dyn_input_ids,
            dyn_position_ids,
            dyn_key_caches,
            dyn_value_caches,
            is_prefill,
            dyn_slot_mapping,
            dynamic_attention_mask,
            dyn_batch_valid_length,
            dyn_q_seq_lens,
            dyn_block_tables,
        )

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        **kwargs
    ):
        key_cache, value_cache = self.get_kvcache()
        attn_metadata = get_forward_context().attn_metadata
        input_ids = input_ids.to(ms.int64)
        if attn_metadata is None:
            attn_metadata = self._dummy_attention_metadata(input_ids, positions)

        if not envs.VLLM_USE_V1:
            seq_lens = attn_metadata.seq_lens
            max_query_len = attn_metadata.max_query_len
            # When Mutli-Step is enabled with Chunked-Prefill, prefills and
            # decodes are scheduled together. In the first step, all the
            # prefills turn into decodes and max_query_len will be 1.
            if self.is_multi_step_chunked_prefill and max_query_len == 1:
                query_lens = [1] * len(seq_lens)
            else:
                query_lens = attn_metadata.query_lens

            seq_lens_np = np.array(seq_lens, dtype=np.int32)
            query_lens_np = np.array(query_lens, dtype=np.int32)
            kv_cache_lens = seq_lens_np - query_lens_np
            is_prefill = attn_metadata.num_decode_tokens == 0 and kv_cache_lens.max() == 0
            slot_mapping = attn_metadata.slot_mapping
            batch_valid_length = Tensor.from_numpy(np.array(attn_metadata.seq_lens, dtype=np.int32))
            q_seq_lens = ms.Tensor(query_lens_np, dtype=ms.int32) if not atlas_inference() else None
            block_tables = attn_metadata.block_tables
            position_ids = ms.Tensor(positions, dtype=ms.int32)
            attn_mask = self.casual_mask.gen_attention_mask(is_prefill, position_ids, query_lens)
        else:
            if attn_metadata.max_context_lens == 0:
                is_prefill = True
            else:
                is_prefill = False
            slot_mapping = attn_metadata.slot_mapping
            seq_lens_np = attn_metadata.seq_lens_np
            batch_valid_length = Tensor.from_numpy(seq_lens_np)
            q_seq_lens = attn_metadata.q_seq_lens if not atlas_inference() else None
            block_tables = attn_metadata.block_tables
            query_lens_np = attn_metadata.q_seq_lens_np
            attn_mask = self.casual_mask.gen_attention_mask(is_prefill, positions, query_lens_np)
            positions = positions.to(ms.int64)
        if is_prefill:
            if not self.enable_high_precision:
                input_ids = ops.expand_dims(input_ids, 0)
            else:
                seq_lens = seq_lens_np.tolist()
                input_ids = _pad_inputs(input_ids, seq_lens, self.config.pad_token_id).reshape(len(seq_lens), -1)
                slot_mapping = _pad_inputs(slot_mapping, seq_lens, -1)
                if atlas_inference():
                    # get pad_token_id form config
                    input_ids = _pad_inputs_using_pfa_mode_length(input_ids, self.config.pad_token_id)
                    slot_mapping = _pad_slot_mapping_using_pfa_mode_length(
                        slot_mapping, input_ids.shape[0]
                    )
            if not self.prefill:
                self.prefill = True
                self.set_model_inputs(self.prefill)
        else:
            input_ids = ops.expand_dims(input_ids, 1)
            if self.prefill:
                self.prefill = False
                self.set_model_inputs(self.prefill)

        # for dummy_attention_metadata
        if is_prefill and not self.set_flags:
            self.set_flags = True

        model_output = self.model(input_ids,
                                  positions,
                                  key_cache,
                                  value_cache,
                                  is_prefill,
                                  slot_mapping,
                                  attn_mask,
                                  batch_valid_length,
                                  q_seq_lens,
                                  block_tables)
        if is_prefill:
            model_output = ops.squeeze(model_output, 0) if not self.enable_high_precision else \
                _unpadding_outputs(model_output, seq_lens_np.tolist())
        else:
            model_output = ops.squeeze(model_output, 1)
        return model_output

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> Set[str]:
        params_dict = self.get_params_dict()
        self.model.load_weights(weights, params_dict)

    def sample(
        self, logits: Tensor, sampling_metadata: SamplingMetadata
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def compute_logits(
        self,
        hidden_states: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    # for v1
    def _dummy_attention_metadata(self, input_ids: Tensor, positions: Tensor):
        input_len = input_ids.shape[0]
        max_seq_len = ms.Tensor(input_len, dtype=ms.int32)
        seq_lengths = ms.Tensor([input_len], dtype=ms.int32)
        q_seq_lens = ms.Tensor([input_len], dtype=ms.int32)
        q_seq_lens_np = np.array([input_len], dtype=np.int32)
        seq_lens_np = np.array([input_len], dtype=np.int32)

        block_tables = ms.Tensor([[0]], dtype=ms.int32)
        slot_mapping = [-1 for _ in range(input_len)]
        slot_mapping = ms.Tensor(slot_mapping, dtype=ms.int32)
        return FlashAttentionMetadata(
            max_seq_len=max_seq_len,
            seq_lens=seq_lengths,
            seq_lens_np=seq_lens_np,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            q_seq_lens=q_seq_lens,
            q_seq_lens_np=q_seq_lens_np,
            context_lens=0,
            # To enforce prefill and decode are both complied in warmup process.
            # So set max_context_lens to 0 for prefill and 1 for decode.
            max_context_lens=0 if not self.set_flags else 1,
            query_start_loc = None
        )
