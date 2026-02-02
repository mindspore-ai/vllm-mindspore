# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_v2.py
#
# Copyright 2025-2026 Huawei Technologites Co., Ltd
# Copyright 2024 The Deepseek team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only DeepSeekV3 model compatible with HuggingFace weights."""
import math
from collections.abc import Iterable
from functools import wraps
from typing import Optional, Union

import numpy as np
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import mint, nn, ops
from transformers import PretrainedConfig
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.sequence import IntermediateTensors

from vllm_mindspore.model_executor.layers.activation import SiluAndMul
from vllm_mindspore.model_executor.layers.fused_moe import FusedMoE
from vllm_mindspore.model_executor.layers.layernorm import RMSNorm
from vllm_mindspore.model_executor.layers.linear import (
    ColumnParallelLinear, MergedColumnParallelLinear, ReplicatedLinear,
    RowParallelLinear)
from vllm_mindspore.model_executor.layers.logits_processor import (
    LogitsProcessor)
from vllm_mindspore.model_executor.layers.rotary_embedding import (
    get_rope, yarn_get_mscale)
from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm_mindspore.model_executor.model_loader.weight_utils import (
    default_weight_loader)
from vllm_mindspore.model_executor.models.interfaces import (MixtureOfExperts,
                                                             SupportsMoeDpTp)
from vllm_mindspore.model_executor.models.model_base import NativeModel
from vllm_mindspore.model_executor.models.utils import (
    is_pp_missing_parameter, make_empty_intermediate_tensors_factory,
    make_layers, maybe_prefix)
from vllm_mindspore.model_executor.utils import (get_model_context,
                                                 set_weight_attrs)

logger = init_logger(__name__)


def transpose_rope_weight(weight, start_dim):
    w1 = weight[..., -start_dim::2, :]
    w2 = weight[..., -start_dim + 1::2, :]
    weight[..., -start_dim:, :] = np.concatenate((w1, w2), axis=-2)
    return weight


def reorder_qkv_rope_proj_weight(weight_loader):

    @wraps(weight_loader)
    def wrapper(*args, **kwargs):
        param = args[-2]
        loaded_weight = args[1][:]
        reorder_params = getattr(param, "reorder_params", {})
        if not reorder_params:
            raise ValueError(
                f"reorder_params of param [{param.name}] should not be empty.")
        qk_rope_head_dim = reorder_params["qk_rope_head_dim"]
        if "kv_head_dim" in reorder_params:
            kv_head_dim = reorder_params["kv_head_dim"]
            loaded_weight = loaded_weight.reshape(kv_head_dim, -1)
        else:
            num_heads = reorder_params["num_heads"]
            q_head_dim = reorder_params["q_head_dim"]
            loaded_weight = loaded_weight.reshape(num_heads, q_head_dim, -1)

        loaded_weight = transpose_rope_weight(loaded_weight, qk_rope_head_dim)
        if "num_heads" in reorder_params:
            loaded_weight = loaded_weight.reshape(num_heads * q_head_dim, -1)
        weight_loader(param, loaded_weight, **kwargs)

    return wrapper


class DeepseekV3MLP(nn.Cell):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            return_bias=False)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           reduce_results=reduce_results,
                                           prefix=f"{prefix}.down_proj",
                                           return_bias=False)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def construct(self,
                  x,
                  dp_pad_index=None,
                  dp_unpad_index=None,
                  dp_pad_index_with_offset=None,
                  dp_unpad_index_total_with_offset=None):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class DeepseekV3MoE(nn.Cell):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.num_redundant_experts = 1
        self.tp_size = get_tensor_model_parallel_world_size()

        self.gate = ReplicatedLinear(
            input_size=config.hidden_size,
            output_size=config.n_routed_experts,
            bias=False,
            prefix=f"{prefix}.gate",
            return_bias=False,
        )
        self.gate.e_score_correction_bias = Parameter(mint.zeros(
            (self.n_routed_experts), dtype=mstype.float32),
                                                      requires_grad=False,
                                                      parallel_optimizer=False)

        self.experts = FusedMoE(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

        if config.n_shared_experts:
            intermediate_size = (config.moe_intermediate_size *
                                 config.n_shared_experts)
            self.shared_experts = DeepseekV3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                reduce_results=self.experts.must_reduce_shared_expert_outputs(
                ),
                prefix=f"{prefix}.shared_experts")

    def construct(self,
                  hidden_states: Tensor,
                  dp_pad_index=None,
                  dp_unpad_index=None,
                  dp_pad_index_with_offset=None,
                  dp_unpad_index_total_with_offset=None):
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_output = self.shared_experts(hidden_states)\
            if self.n_shared_experts else None

        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            dp_pad_index=dp_pad_index,
            dp_unpad_index=dp_unpad_index,
            dp_pad_index_with_offset=dp_pad_index_with_offset,
            dp_unpad_index_total_with_offset=dp_unpad_index_total_with_offset)
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        if self.tp_size > 1:
            final_hidden_states = self.experts.\
                maybe_all_reduce_tensor_model_parallel(
                    final_hidden_states=final_hidden_states, )
        return final_hidden_states.view(orig_shape)


class DeepseekV3Attention(nn.Cell):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.local_num_heads = self.total_num_heads // self.tp_size
        self.head_dim = config.head_dim if hasattr(config, "head_dim") else \
            self.hidden_size // self.total_num_heads

        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.rope_scaling = config.rope_scaling

        if self.rope_scaling:
            self.rope_scaling["rope_type"] = "deepseek_yarn"
        self.rotary_emb = get_rope(
            head_size=self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=self.rope_scaling,
        )

        self.scaling = 1. / math.sqrt(self.q_head_dim)
        if self.rope_scaling:
            mscale_all_dim = self.rope_scaling.get("mscale_all_dim", 1)
            scaling_factor = self.rope_scaling.get("factor", 1)
            mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
            self.scaling = self.scaling * mscale * mscale

        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()
        self.flash_attention = ops.operations.nn_ops.FlashAttentionScore(
            head_num=self.local_num_heads,
            scale_value=self.scaling,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            input_layout="TH")
        self.paged_attention = ops.auto_generate.PagedAttention(
            head_num=self.local_num_heads,
            scale_value=self.scaling,
            kv_head_num=1,
            mla_v_dim=self.kv_lora_rank)

        self.q_a_proj = ReplicatedLinear(self.hidden_size,
                                         self.q_lora_rank,
                                         bias=False,
                                         quant_config=quant_config,
                                         return_bias=False,
                                         prefix=f"{prefix}.q_a_proj")

        self.q_a_layernorm = RMSNorm(self.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(self.q_lora_rank,
                                             self.total_num_heads *
                                             self.q_head_dim,
                                             bias=False,
                                             quant_config=quant_config,
                                             return_bias=False,
                                             prefix=f"{prefix}.q_b_proj")
        # q_rope weight transpose in advance
        # so that the transpose operator can be omitted
        set_weight_attrs(
            self.q_b_proj.weight, {
                "reorder_params": {
                    "qk_rope_head_dim": self.qk_rope_head_dim,
                    "num_heads": self.total_num_heads,
                    "q_head_dim": self.q_head_dim,
                },
                "weight_loader":
                reorder_qkv_rope_proj_weight(self.q_b_proj.weight_loader)
            })

        # 1. kv_a_proj_with_mqa: kv latent vector;
        # 2. kv_a_layernorm: latent vector of kv normalization
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.kv_a_proj_with_mqa")
        # the same operation to the q_rope weight
        set_weight_attrs(
            self.kv_a_proj_with_mqa.weight, {
                "reorder_params": {
                    "qk_rope_head_dim": self.qk_rope_head_dim,
                    "kv_head_dim": self.kv_head_dim,
                },
                "weight_loader":
                reorder_qkv_rope_proj_weight(
                    self.kv_a_proj_with_mqa.weight_loader)
            })

        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj_k = ColumnParallelLinear(self.kv_lora_rank,
                                                self.total_num_heads *
                                                self.qk_nope_head_dim,
                                                bias=False,
                                                quant_config=quant_config,
                                                return_bias=False,
                                                prefix=f"{prefix}.kv_b_proj_k")

        self.kv_b_proj_v = ColumnParallelLinear(self.kv_lora_rank,
                                                self.total_num_heads *
                                                self.v_head_dim,
                                                bias=False,
                                                quant_config=quant_config,
                                                return_bias=False,
                                                prefix=f"{prefix}.kv_b_proj_v")

        self.o_proj = RowParallelLinear(self.total_num_heads * self.v_head_dim,
                                        self.hidden_size,
                                        bias=False,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj",
                                        return_bias=False)

        self.reshape = ops.Reshape()
        self.tile_kv = ops.Tile()
        self.dim_slice_4d = ops.Slice()
        self.kpe_concat = ops.Concat(1)
        self.pe_concat = ops.Concat(2)
        self.qabsorb_k_matmul = ops.BatchMatMul()
        self.outabsorb_v_matmul = ops.BatchMatMul(transpose_b=True)
        self.split = ops.auto_generate.SplitWithSize()

    def forward_absorb_prepare(
        self,
        hidden_states,
        positions: Tensor,
        key_cache: Tensor,
        slot_mapping: Tensor,
        batch_valid_length: Tensor,
        is_prefill: bool,
    ):
        # calculate q
        q = self.q_a_proj(hidden_states)
        norm_q = self.q_a_layernorm(q)
        q = self.q_b_proj(norm_q)
        q = q.view((-1, self.local_num_heads, self.q_head_dim))

        # calculate k(v)
        latent_kv_all = self.kv_a_proj_with_mqa(hidden_states)
        latent_kv, k_pe = self.split(
            latent_kv_all, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        i_kv = self.kv_a_layernorm(latent_kv)

        # q， k rope
        q_nope, q_pe = self.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = q_pe.view((-1, self.local_num_heads * self.qk_rope_head_dim))
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe, batch_valid_length,
                                     is_prefill)
        q_pe = q_pe.view((-1, self.local_num_heads, self.qk_rope_head_dim))

        # k reshape_and_cache
        key_states_cache = mint.cat((i_kv, k_pe), 1)
        key_out = self.reshape_and_cache(key_states_cache, None, key_cache,
                                         None, slot_mapping)
        q_nope = ops.depend(q_nope, key_out)

        return q_nope, q_pe, k_pe, i_kv

    def construct(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        key_cache: Tensor,
        slot_mapping: Tensor,
        attn_mask: Tensor,
        batch_valid_length: Tensor,
        q_seq_lens: Tensor,
        block_tables: Tensor,
    ) -> Tensor:
        is_prefill = get_model_context("is_prefill")
        q_nope, q_pe, k_pe, i_kv = self.forward_absorb_prepare(
            hidden_states=hidden_states,
            positions=positions,
            key_cache=key_cache,
            slot_mapping=slot_mapping,
            batch_valid_length=batch_valid_length,
            is_prefill=is_prefill,
        )

        if is_prefill:
            # q
            query_states = mint.cat((q_nope, q_pe), 2)

            # k
            k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)
            k_pe = self.tile_kv(k_pe, (1, self.local_num_heads, 1))
            o_k_nope = self.kv_b_proj_k(i_kv)
            k_nope = o_k_nope.view(-1, self.local_num_heads,
                                   self.qk_nope_head_dim)
            key_states = self.pe_concat((k_nope, k_pe))

            # v
            o_v = self.kv_b_proj_v(i_kv)
            value_states = o_v.view(
                (-1, self.local_num_heads, self.v_head_dim))
            # It's not necessary.
            # Just fa is not support k != v. V just (t, head, 128)
            value_states = self.pe_concat((value_states, k_pe))

            # attention
            query_states = query_states.view(
                -1, self.local_num_heads * self.q_head_dim)
            key_states = key_states.view(
                -1, self.local_num_heads * self.q_head_dim)
            value_states = value_states.view(
                -1, self.local_num_heads * self.q_head_dim)
            context_layer = self.flash_attention(
                query_states,
                key_states,
                value_states,
                None,
                None,
                None,
                attn_mask,
                None,
                actual_seq_qlen=q_seq_lens,
                actual_seq_kvlen=batch_valid_length,
            )[-1]
            context_layer = context_layer.view(-1, self.local_num_heads,
                                               self.q_head_dim)
            context_layer = self.dim_slice_4d(
                context_layer, (0, 0, 0),
                (-1, self.local_num_heads, self.v_head_dim))
        else:
            # q,  k_absorb
            q_absorb = self.kv_b_proj_k.weight.view(self.local_num_heads,
                                                    self.qk_nope_head_dim,
                                                    self.kv_lora_rank)
            q_nope = self.qabsorb_k_matmul(q_nope.transpose(1, 0, 2),
                                           q_absorb).transpose(1, 0, 2)
            query_states = self.pe_concat((q_nope, q_pe))
            query_states = query_states.view(
                -1,
                self.local_num_heads *
                (self.kv_lora_rank + self.qk_rope_head_dim))

            # attention
            context_layer = self.paged_attention(
                query_states,
                key_cache,
                key_cache,
                block_tables,
                batch_valid_length,
                None,
                None,
                attn_mask,
                q_seq_lens,
            )
            context_layer = context_layer.view(-1, self.local_num_heads,
                                               self.kv_lora_rank)

            # out, v_absorb
            out_absorb = self.kv_b_proj_v.weight.view(self.local_num_heads,
                                                      self.v_head_dim,
                                                      self.kv_lora_rank)
            context_layer = self.outabsorb_v_matmul(
                context_layer.transpose(1, 0, 2),
                out_absorb).transpose(1, 0, 2)

        attn_out = context_layer.view(-1,
                                      self.local_num_heads * self.v_head_dim)
        output = self.o_proj(attn_out)
        return output


class DeepseekV3DecoderLayer(nn.Cell):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_idx = int(prefix.split(sep='.')[-1])
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV3Attention(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        if (config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0):
            self.mlp = DeepseekV3MoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp")

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def construct(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        key_cache: Tensor,
        slot_mapping: Tensor,
        attn_mask: Tensor,
        batch_valid_length: Tensor,
        q_seq_lens: Tensor,
        block_tables: Tensor,
        residual: Tensor,
        dp_pad_index: Optional[bool] = None,
        dp_unpad_index: Optional[Tensor] = None,
        dp_pad_index_with_offset: Optional[Tensor] = None,
        dp_unpad_index_total_with_offset: Optional[Tensor] = None,
    ) -> Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(hidden_states, positions, key_cache,
                                       slot_mapping, attn_mask,
                                       batch_valid_length, q_seq_lens,
                                       block_tables)

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        hidden_states = self.mlp(hidden_states, dp_pad_index, dp_unpad_index,
                                 dp_pad_index_with_offset,
                                 dp_unpad_index_total_with_offset)

        return hidden_states, residual


class DeepseekV3Model(nn.Cell):

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens")

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV3DecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids) -> Tensor:
        return self.embed_tokens(input_ids)

    def construct(
        self,
        input_ids: Tensor,
        positions: Tensor,
        key_caches: list[Tensor],
        slot_mapping: Tensor,
        attn_mask: Tensor,
        batch_valid_length: Tensor,
        q_seq_lens: Tensor,
        block_tables: Tensor,
        intermediate_hidden_states: Optional[Tensor] = None,
        intermediate_residual: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        dp_pad_index: Optional[Tensor] = None,
        dp_unpad_index: Optional[Tensor] = None,
        dp_pad_index_total_with_offset: Optional[Tensor] = None,
        dp_unpad_index_total_with_offset: Optional[Tensor] = None,
    ) -> Union[Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_hidden_states is not None
            hidden_states = intermediate_hidden_states
            residual = intermediate_residual

        for i in range(self.start_layer, self.end_layer):
            layer: DeepseekV3DecoderLayer = self.layers[i]
            hidden_states, residual = layer(
                hidden_states=hidden_states,
                positions=positions,
                key_cache=key_caches[i - self.start_layer],
                slot_mapping=slot_mapping,
                attn_mask=attn_mask,
                batch_valid_length=batch_valid_length,
                q_seq_lens=q_seq_lens,
                block_tables=block_tables,
                residual=residual,
                dp_pad_index=dp_pad_index,
                dp_unpad_index=dp_unpad_index,
                dp_pad_index_with_offset=dp_pad_index_total_with_offset,
                dp_unpad_index_total_with_offset=
                dp_unpad_index_total_with_offset)
        if get_pp_group().is_last_rank:
            hidden_states, residual = self.norm(hidden_states, residual)
        return hidden_states, residual

    def load_weights(self, weights: Iterable[tuple[str, Tensor]],
                     params_dict: dict[str, Parameter]):
        stacked_params_mapping = [
            # the format is (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # the format is (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts)

        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if "kv_b_proj" in name and name not in params_dict:
                loaded_weight = loaded_weight[:]
                k_name = name.replace("kv_b_proj", "kv_b_proj_k")
                v_name = name.replace("kv_b_proj", "kv_b_proj_v")

                loaded_weight = loaded_weight.reshape(
                    self.config.num_attention_heads,
                    self.config.qk_nope_head_dim + self.config.v_head_dim, -1)
                k_weight = loaded_weight[:, :self.config.
                                         qk_nope_head_dim, :].reshape(
                                             self.config.num_attention_heads *
                                             self.config.qk_nope_head_dim, -1)
                v_weight = loaded_weight[:, self.config.
                                         qk_nope_head_dim:, :].reshape(
                                             self.config.num_attention_heads *
                                             self.config.qk_nope_head_dim, -1)

                if k_name not in params_dict or v_name not in params_dict:
                    continue
                k_param = params_dict[k_name]
                v_param = params_dict[v_name]

                k_param.weight_loader(k_param, k_weight)
                v_param.weight_loader(v_param, v_weight)
                loaded_params.add(k_name)
                loaded_params.add(v_name)
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if (("mlp.experts." in name) and name not in params_dict):
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True

                    # Do not modify `name` since the loop may continue here
                    # Instead, create a new variable
                    name_mapped = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name_mapped, self):
                        continue

                    if name_mapped not in params_dict:
                        continue

                    param = params_dict[name_mapped]
                    weight_loader = param.weight_loader

                    weight_loader(param,
                                  loaded_weight,
                                  name_mapped,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    loaded_params.add(name_mapped)
                    break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue

                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    # Remapping the name of FP8 kv-scale.
                    if name is None:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    if name not in params_dict:
                        logger.warning(
                            f"name_mapped: {name} not found in params_dict"  # noqa: G004
                        )
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
        return loaded_params


class DeepseekV3ForCausalLM(NativeModel, SupportsMoeDpTp, MixtureOfExperts):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = DeepseekV3Model(vllm_config=vllm_config,
                                     prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
        self.expert_weights = []

        self.common_preprocess(vllm_config, prefix=prefix)

        self.dp_pad_input = False

        self.enable_expert_parallel = False

    def get_input_embeddings(self, input_ids: Tensor) -> Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(self,
                input_ids: Tensor,
                positions: Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[Tensor] = None,
                **kwargs) -> Union[Tensor, IntermediateTensors]:
        hidden_states, residual = self.exec_model(input_ids, positions,
                                                  intermediate_tensors,
                                                  inputs_embeds)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        return hidden_states

    def compute_logits(self, hidden_states: Tensor) -> Optional[Tensor]:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        params_dict = self.get_params_dict()
        return self.model.load_weights(weights, params_dict)
