# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/minicpm.py
# Copyright 2025 Huawei Technologies Co., Ltd.
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

# The adaptation of minicpm in vllm-mindspore mainly includes the
# following points:
# 1. Additional model input parameters have been added, such as
#    `key_cache`, `block_tables`, etc., to accommodate the
#    vllm-mindspore calling convention.
# 2. During model initialization, methods from the NativeModel base
#    class, such as `common_preprocess`, are invoked to adapt to the
#    vllm-mindspore workflow.
# 3. In the `forward` function, the `exec_model` method is called to
#    perform the model’s forward computation, aligning with the
#    vllm-mindspore execution flow.
# 4. In the `load_weights` function, due to the lack of `skip_prefix`
#    functionality, the handling of `tie_word_embeddings` has been
#    adapted.
# SPDX-License-Identifier: Apache-2.0
"""Inference-only MiniCPM model compatible with HuggingFace weights."""
import math
from collections.abc import Iterable
from typing import Any, Optional, Union

import mindspore as ms
from mindspore import Tensor, nn, ops
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.sequence import IntermediateTensors

from vllm_mindspore.model_executor.layers.activation import (FatreluAndMul,
                                                             SiluAndMul)
from vllm_mindspore.model_executor.layers.layernorm import RMSNorm
from vllm_mindspore.model_executor.layers.linear import (
    MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear)
from vllm_mindspore.model_executor.layers.logits_processor import (
    LogitsProcessor)
from vllm_mindspore.model_executor.layers.rotary_embedding import get_rope
from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm_mindspore.model_executor.model_loader.weight_utils import (
    default_weight_loader)
from vllm_mindspore.model_executor.models.model_base import NativeModel
from vllm_mindspore.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)
from vllm_mindspore.v1.attention import Attention


class MiniCPMMLP(nn.Cell):
    """MindSpore version of MLP implementation"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        hidden_act_param: float,
        quant_config=None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        if hidden_act == "silu":
            self.act_fn = SiluAndMul()
        elif hidden_act == "fatrelu":
            self.act_fn = FatreluAndMul(threshold=hidden_act_param)
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu and fatrelu are supported for now.")

    def construct(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class MiniCPMAttention(nn.Cell):
    """MindSpore version of Attention implementation"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config=None,
        quant_config=None,
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
        self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )

        self.split_op = ops.auto_generate.SplitWithSize()

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

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
        q, k, v = self.split_op(qkv, [self.q_size, self.kv_size, self.kv_size],
                                dim=-1)
        orig_dtype = q.dtype
        q, k = q.astype(ms.float32), k.astype(ms.float32)
        q, k = self.rotary_emb(positions, q, k, batch_valid_length)
        q, k = q.astype(orig_dtype), k.astype(orig_dtype)
        attn_output = self.attn(q, k, v, key_cache, value_cache, slot_mapping,
                                attn_mask, batch_valid_length, q_seq_lens,
                                block_tables)
        output, _ = self.o_proj(attn_output)
        return output


class MiniCPMDecoderLayer(nn.Cell):
    """MindSpore version of Decoder layer implementation"""

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.hidden_size = config.hidden_size
        self.rope_theta = getattr(config, "rope_theta", 10000)
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.max_position_embeddings = getattr(config,
                                               "max_position_embeddings", 8192)
        self.prefix = prefix

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.self_attn = MiniCPMAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
            max_position_embeddings=self.max_position_embeddings,
            cache_config=self.cache_config,
            quant_config=self.quant_config,
            prefix=f"{prefix}.self_attn",
        )

        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.num_experts = getattr(self.config, "num_experts", 0)
        if self.num_experts == 0:
            self.mlp = MiniCPMMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                hidden_act_param=getattr(config, "hidden_act_param", 0.),
                quant_config=quant_config,
            )
        else:
            raise ValueError("Unsupported model structure: MiniCPMMoE. "
                             "Only MiniCPMMLP is supported for now.")

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
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, key_cache,
                                       value_cache, slot_mapping, attn_mask,
                                       batch_valid_length, q_seq_lens,
                                       block_tables)
        hidden_states = residual + hidden_states * (
            self.config.scale_depth / math.sqrt(self.config.num_hidden_layers))

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * \
            (self.config.scale_depth / math.sqrt(self.config.num_hidden_layers))

        return hidden_states, None


class MiniCPMModel(nn.Cell):
    """MindSpore version of MiniCPM model implementation"""

    def __init__(
        self,
        *,
        vllm_config,
        prefix: str = "",
        layer_type: type[MiniCPMDecoderLayer] = MiniCPMDecoderLayer,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config  # noqa: F841

        self.config = config
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=quant_config,
        )
        self.num_experts = getattr(config, "num_experts", 0)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.make_empty_intermediate_tensors = \
            make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size)

    def get_input_embeddings(self, input_ids: Tensor) -> Tensor:
        embedding = self.embed_tokens(input_ids)
        return embedding * self.config.scale_emb

    def construct(
        self,
        input_ids: Optional[Tensor],
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
    ) -> Union[Tensor, IntermediateTensors]:
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

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, Tensor]], params_dict):
        loaded_params: set[str] = set()
        stacked_params_mapping = [
            # shape is (param_name, shard_name, shard_id).
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        expert_params_mapping = [
            # shape is (param_name, shard_name, shard_id).
            ("ws" if weight_name in ["w1", "w3"] else "w2s",
             f"experts.{expert_id}.{weight_name}.weight", expert_id)
            for expert_id in range(self.num_experts)
            for weight_name in ["w1", "w2", "w3"]
        ]

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or \
                "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(name)
                    break
            else:
                for param_name, weight_name, expert_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  weight_name,
                                  expert_id=expert_id)
                    loaded_params.add(name)
                    break

                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        return loaded_params


class MiniCPMForCausalLM(NativeModel):
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

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    """MindSpore version of MiniCPM for causal language modeling"""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.prefix = prefix
        self.vllm_config = vllm_config
        self.config = config
        self.lora_config = lora_config
        self.cache_config = cache_config
        self.quant_config = quant_config

        self.model = MiniCPMModel(vllm_config=vllm_config,
                                  prefix=maybe_prefix(prefix, "model"))

        unpadded_vocab_size = config.vocab_size
        if lora_config:
            unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=(DEFAULT_VOCAB_PADDING_SIZE if not lora_config else
                          lora_config.lora_vocab_padding_size),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        self.scale_width = self.config.hidden_size / self.config.dim_model_base
        self.logits_processor = LogitsProcessor(unpadded_vocab_size,
                                                config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        self.set_modules({"model": self.model, "lm_head": self.lm_head})

        self.common_preprocess(vllm_config, prefix)

    def forward(self,
                input_ids,
                positions,
                intermediate_tensors=None,
                inputs_embeds=None,
                **kwargs):
        hidden_states = self.exec_model(input_ids, positions,
                                        intermediate_tensors, inputs_embeds)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        params_dict = self.get_params_dict()
        load_params = self.model.load_weights(weights, params_dict)
        if self.config.tie_word_embeddings:
            load_params.add("lm_head.weight")
        return load_params

    def compute_logits(
        self,
        hidden_states: Tensor,
    ) -> Optional[Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits
