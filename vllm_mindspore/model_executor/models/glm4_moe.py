# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/glm4_moe.py
#
# Copyright 2026 Huawei Technologies Co., Ltd
# Copyright 2025 The ZhipuAI Team.
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
"""Inference-only GLM-4 MoE model compatible with HuggingFace weights."""
from collections.abc import Iterable
from typing import Any, Optional, Union

from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import mint, nn
from transformers.models.glm4_moe import Glm4MoeConfig
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.sequence import IntermediateTensors

from vllm_mindspore.model_executor.layers.activation import SiluAndMul
from vllm_mindspore.model_executor.layers.fused_moe import FusedMoE
from vllm_mindspore.model_executor.layers.layernorm import RMSNorm
from vllm_mindspore.model_executor.layers.linear import (
    MergedColumnParallelLinear, QKVParallelLinear, ReplicatedLinear,
    RowParallelLinear)
from vllm_mindspore.model_executor.layers.logits_processor import (
    LogitsProcessor)
from vllm_mindspore.model_executor.layers.rotary_embedding import get_rope
from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm_mindspore.model_executor.model_loader.weight_utils import (
    default_weight_loader)
from vllm_mindspore.model_executor.models.model_base import NativeModel
from vllm_mindspore.model_executor.models.utils import (
    PPMissingLayer, extract_layer_index, is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)
from vllm_mindspore.v1.attention import Attention

logger = init_logger(__name__)


class Glm4MoeMLP(nn.Cell):
    """GLM-4 MoE MLP layer (for dense layers)."""

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
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           reduce_results=reduce_results,
                                           prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def construct(self, x: Tensor) -> Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Glm4Moe(nn.Cell):
    """
    GLM-4 MoE Sparse MoE Block.
    Reference: vllm_mindspore/model_executor/models/qwen3_moe.py
    """

    def __init__(
        self,
        config: Glm4MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}.")

        # Router gate - use ReplicatedLinear instead of nn.Linear
        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.n_routed_experts,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")

        self.gate.e_score_correction_bias = Parameter(mint.zeros(
            (config.n_routed_experts, ), dtype=mstype.float32),
                                                      requires_grad=False,
                                                      parallel_optimizer=False)

        # Expert network using FusedMoE with sigmoid scoring function
        self.experts = FusedMoE(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func="sigmoid",
        )

        # Shared experts (if configured)
        if config.n_shared_experts is not None:
            intermediate_size = (config.moe_intermediate_size *
                                 config.n_shared_experts)
            self.shared_experts = Glm4MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

    def construct(self, hidden_states: Tensor) -> Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Get router logits
        router_logits, _ = self.gate(hidden_states)

        # Routed expert output
        routed_output = self.experts(hidden_states=hidden_states,
                                     router_logits=router_logits)

        # Shared expert output
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            final_hidden_states = (routed_output * self.routed_scaling_factor +
                                   shared_output)
        else:
            final_hidden_states = routed_output * self.routed_scaling_factor

        # All-reduce for tensor parallelism
        if self.tp_size > 1:
            final_hidden_states = \
                self.experts.maybe_all_reduce_tensor_model_parallel(
                    final_hidden_states)

        return final_hidden_states.view(orig_shape)


class Glm4MoeAttention(nn.Cell):
    """GLM-4 MoE Attention layer with optional QK-Norm."""

    def __init__(
        self,
        config: Glm4MoeConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 131072,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-05,
        qkv_bias: bool = False,
        use_qk_norm: bool = False,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
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
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.use_qk_norm = use_qk_norm

        self.qkv_proj = QKVParallelLinear(hidden_size,
                                          self.head_dim,
                                          self.total_num_heads,
                                          self.total_num_kv_heads,
                                          bias=qkv_bias,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.qkv_proj")

        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim,
                                        hidden_size,
                                        bias=False,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")

        partial_rotary_factor = getattr(config, "partial_rotary_factor", 0.5)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            partial_rotary_factor=partial_rotary_factor,
        )

        # Attention layer - use vllm_mindspore Attention
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        # QK-Norm layers
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def construct(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        slot_mapping: Tensor,
        attn_mask: Tensor,
        batch_valid_length: Tensor,
        q_seq_lens: Tensor,
        block_tables: Tensor,
    ) -> Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.use_qk_norm:
            # Apply QK-Norm
            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                               self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            q = q_by_head.view(q.shape)

            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                               self.head_dim)
            k_by_head = self.k_norm(k_by_head)
            k = k_by_head.view(k.shape)

        q, k = self.rotary_emb(positions, q, k, batch_valid_length)

        # Attention computation with all KV cache parameters
        attn_output = self.attn(q, k, v, key_cache, value_cache, slot_mapping,
                                attn_mask, batch_valid_length, q_seq_lens,
                                block_tables)
        output, _ = self.o_proj(attn_output)
        return output


class Glm4MoeDecoderLayer(nn.Cell):
    """GLM-4 MoE Decoder Layer with auto Dense/MoE selection."""

    def __init__(
        self,
        config: Glm4MoeConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          131072)
        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx

        self.self_attn = Glm4MoeAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=config.attention_bias,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            use_qk_norm=config.use_qk_norm,
        )

        # Select Dense MLP or MoE based on layer index
        if (config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace):
            self.mlp = Glm4Moe(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Glm4MoeMLP(hidden_size=config.hidden_size,
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
        positions: Tensor,
        hidden_states: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        slot_mapping: Tensor,
        attn_mask: Tensor,
        batch_valid_length: Tensor,
        q_seq_lens: Tensor,
        block_tables: Tensor,
        residual: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states, key_cache,
                                       value_cache, slot_mapping, attn_mask,
                                       batch_valid_length, q_seq_lens,
                                       block_tables)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Glm4MoeModel(nn.Cell):
    """GLM-4 MoE Transformer Model."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens")
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Glm4MoeDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers")

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: Tensor) -> Tensor:
        return self.embed_tokens(input_ids)

    def construct(
        self,
        input_ids: Tensor,
        positions: Tensor,
        key_caches: list[Tensor],
        value_caches: list[Tensor],
        slot_mapping: Tensor,
        attn_mask: Tensor,
        batch_valid_length: Tensor,
        q_seq_lens: Tensor,
        block_tables: Tensor,
        intermediate_hidden_states: Optional[Tensor] = None,
        intermediate_residual: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            hidden_states = intermediate_hidden_states
            residual = intermediate_residual

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states,
                                            key_caches[i - self.start_layer],
                                            value_caches[i - self.start_layer],
                                            slot_mapping, attn_mask,
                                            batch_valid_length, q_seq_lens,
                                            block_tables, residual)

        if get_pp_group().is_last_rank:
            hidden_states, residual = self.norm(hidden_states, residual)

        return hidden_states, residual

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts)

    def load_weights(self, weights: Iterable[tuple[str, Tensor]],
                     params_dict: dict[str, Parameter]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)  # noqa: ERA001
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = self.get_expert_mapping()
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
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
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if ((name.endswith(".bias") or name.endswith("_bias"))
                        and name not in params_dict):
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class Glm4MoeForCausalLM(NativeModel):
    """GLM-4 MoE CausalLM entry class."""
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = Glm4MoeModel(vllm_config=vllm_config,
                                  prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size,
                                          quant_config=quant_config,
                                          prefix=maybe_prefix(
                                              prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        # Call common preprocessing
        self.common_preprocess(vllm_config, prefix)

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

    def compute_logits(
        self,
        hidden_states: Tensor,
    ) -> Optional[Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        params_dict = self.get_params_dict()
        return self.model.load_weights(weights, params_dict)
