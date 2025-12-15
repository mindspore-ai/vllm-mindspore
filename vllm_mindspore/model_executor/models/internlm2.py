# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/internlm2.py
#
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
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

# The adaptation of internlm2 in vllm-mindspore mainly includes the
# following points:
# 1. Additional model input parameters have been added, such as
#    `key_cache`, `block_tables`, etc., to accommodate the
#    vllm-mindspore calling convention.
# 2. During model initialization, methods from the NativeModel base
#    class, such as `common_preprocess`, are invoked to adapt to the
#    vllm-mindspore workflow.
# 3. In the `forward` function, the `exec_model` method is called to
#    perform the model's forward computation, aligning with the
#    vllm-mindspore execution flow.
# 4. In the `load_weights` function, due to the lack of `skip_prefix`
#    functionality, the handling of `tie_word_embeddings` has been
#    adapted.
"""Inference-only InternLM2 model compatible with HuggingFace weights."""
from collections.abc import Iterable
from functools import partial
from typing import Any, Optional, Union

from mindspore import Parameter, Tensor, mint, nn
from transformers import PretrainedConfig
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm_mindspore.attention.layer import Attention
from vllm_mindspore.distributed.communication_op import (
    AllGatherFromModelParallelRegion)
from vllm_mindspore.model_executor.layers.activation import SiluAndMul
from vllm_mindspore.model_executor.layers.layernorm import RMSNorm
from vllm_mindspore.model_executor.layers.linear import (
    MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear)
from vllm_mindspore.model_executor.layers.logits_processor import (
    LogitsProcessor)
from vllm_mindspore.model_executor.layers.rotary_embedding import get_rope
from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm_mindspore.model_executor.model_loader.weight_utils import (
    default_weight_loader)
from vllm_mindspore.model_executor.models.model_base import NativeModel
from vllm_mindspore.model_executor.models.utils import (
    PPMissingLayer, is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)

logger = init_logger(__name__)


def split_tensor_along_last_dim(
    tensor: Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
):
    """Split a tensor along its last dimension.

    Args:
        tensor: Input tensor to split.
        num_partitions: Number of partitions to split the tensor into.
        contiguous_split_chunks: If True, make each chunk contiguous in memory.

    Returns:
        A list of tensors split along the last dimension.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.shape[last_dim] // num_partitions
    # Split.
    tensor_list = mint.split(tensor, [last_dim_size] * num_partitions,
                             dim=last_dim)

    # NOTE: mint.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class InternLM2MLP(nn.Cell):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.w2 = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w2",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def construct(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.w2(x)
        return x


class InternLM2Attention(nn.Cell):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.total_num_heads = num_heads
        assert self.total_num_heads % self.tp_size == 0
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= self.tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.key_value_groups = int(self.num_heads / self.num_kv_heads)
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.wqkv = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wqkv",
        )
        self.wo = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wo",
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

    def split_qkv(self, qkv: Tensor):
        """Split the query, key, and value tensors.
        
        Args:
            qkv: The combined query, key, and value tensor.
            
        Returns:
            Tuple of split query, key, and value tensors.
        """
        seq_len = qkv.shape[0]
        if self.tp_size > 1:
            all_gather = AllGatherFromModelParallelRegion()
            qkv_map = [self.q_size, self.kv_size, self.kv_size] * self.tp_size
            qkv = all_gather(qkv)
            qkv = mint.split(qkv, qkv_map, dim=-1)
            qkv = qkv[::3] + qkv[1::3] + qkv[2::3]
            qkv = mint.cat(qkv, dim=-1)

        qkv = qkv.view(seq_len, self.total_num_kv_heads,
                       self.key_value_groups + 2, self.head_dim)
        q, k, v = mint.split(qkv, [self.key_value_groups, 1, 1], dim=-2)
        q = q.reshape(seq_len, self.q_size * self.tp_size)
        k = k.reshape(seq_len, self.kv_size * self.tp_size)
        v = v.reshape(seq_len, self.kv_size * self.tp_size)

        if self.tp_size > 1:
            splitter = partial(split_tensor_along_last_dim,
                               num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
            v = splitter(v)[self.tp_rank]
        return q, k, v

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
        """Forward pass for the attention layer.
        
        Args:
            positions: Positional indices for the input tokens.
            hidden_states: Input hidden states.
            key_cache: Cached keys for attention computation.
            value_cache: Cached values for attention computation.
            slot_mapping: Mapping of slots for PagedAttention.
            attn_mask: Attention mask.
            batch_valid_length: Valid sequence lengths for the batch.
            q_seq_lens: Query sequence lengths.
            block_tables: Block tables for PagedAttention.
            
        Returns:
            Output tensor after attention computation.
        """
        qkv, _ = self.wqkv(hidden_states)
        q, k, v = self.split_qkv(qkv)
        q, k = self.rotary_emb(positions, q, k, batch_valid_length)
        attn_output = self.attn(q, k, v, key_cache, value_cache, slot_mapping,
                                attn_mask, batch_valid_length, q_seq_lens,
                                block_tables)
        output, _ = self.wo(attn_output)
        return output


class InternLMDecoderLayer(nn.Cell):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.attention = InternLM2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attention",
        )
        self.feed_forward = InternLM2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.feed_forward",
        )
        self.attention_norm = RMSNorm(config.hidden_size,
                                      eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        """Forward pass for the decoder layer.
        
        Args:
            positions: Positional indices for the input tokens.
            hidden_states: Input hidden states.
            key_cache: Cached keys for attention computation.
            value_cache: Cached values for attention computation.
            slot_mapping: Mapping of slots for PagedAttention.
            attn_mask: Attention mask.
            batch_valid_length: Valid sequence lengths for the batch.
            q_seq_lens: Query sequence lengths.
            block_tables: Block tables for PagedAttention.
            residual: Residual connection from the previous layer.
            
        Returns:
            Tuple of output hidden states and residual for the next layer.
        """
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.attention_norm(hidden_states)
        else:
            hidden_states, residual = self.attention_norm(
                hidden_states, residual)
        hidden_states = self.attention(
            positions=positions,
            hidden_states=hidden_states,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            batch_valid_length=batch_valid_length,
            q_seq_lens=q_seq_lens,
            block_tables=block_tables,
        )

        # Fully Connected
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


class InternLM2Model(nn.Cell):

    def __init__(
            self,
            *,
            vllm_config: VllmConfig,
            prefix: str = "",
            layer_type: type[InternLMDecoderLayer] = InternLMDecoderLayer):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size
        if get_pp_group().is_first_rank:
            self.tok_embeddings = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.tok_embeddings = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(
                config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.layers")
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: Tensor) -> Tensor:
        return self.tok_embeddings(input_ids)

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
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[Tensor] = None,
    ) -> Union[Tensor, IntermediateTensors]:
        """Forward pass for the InternLM2 model.
        
        Args:
            input_ids: Input token IDs.
            positions: Positional indices for the input tokens.
            key_caches: List of cached keys for attention computation.
            value_caches: List of cached values for attention computation.
            slot_mapping: Mapping of slots for PagedAttention.
            attn_mask: Attention mask.
            batch_valid_length: Valid sequence lengths for the batch.
            q_seq_lens: Query sequence lengths.
            block_tables: Block tables for PagedAttention.
            intermediate_tensors: Intermediate tensors for pipeline parallelism.
            inputs_embeds: Pre-computed input embeddings.
            
        Returns:
            Output hidden states or intermediate tensors for pipeline
            parallelism.
        """
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

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

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, Tensor]],
                     params_dict: dict[str, Parameter]) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
        ]
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class InternLM2ForCausalLM(NativeModel):
    packed_modules_mapping = {
        "wqkv": ["wqkv"],
        "gate_up_proj": ["w1", "w3"],
    }

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 model_type: type[InternLM2Model] = InternLM2Model):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.quant_config = quant_config
        self.lora_config = lora_config

        self.model = model_type(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.output = ParallelLMHead(config.vocab_size,
                                         config.hidden_size,
                                         quant_config=quant_config,
                                         prefix=maybe_prefix(
                                             prefix, "lm_head"))
            if self.config.tie_word_embeddings:
                self.output.weight = self.model.tok_embeddings.weight
            self.sampler = get_sampler()
        else:
            self.output = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        self.common_preprocess(vllm_config, prefix)
        self.set_modules({"model": self.model, "output": self.output})

    def get_input_embeddings(self, input_ids: Tensor) -> Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(self,
                input_ids: Tensor,
                positions: Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[Tensor] = None,
                **kwargs) -> Union[Tensor, IntermediateTensors]:
        hidden_states = self.exec_model(input_ids, positions,
                                        intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[Tensor]:
        logits = self.logits_processor(self.output, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        params_dict = self.get_params_dict()
        self.model.load_weights(weights, params_dict)

    def sample(self, logits: Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    @property
    def ready_lm_head(self) -> nn.Cell:
        if self.output is None:
            raise RuntimeError("lm head not initialized")
        return self.output
