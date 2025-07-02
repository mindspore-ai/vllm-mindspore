#!/usr/bin/env python3
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
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from transformers import Qwen3Config

import numpy as np
from mindone.transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention, Qwen3MLP, Qwen3PreTrainedModel, Qwen3RMSNorm)
from mindspore import Tensor, jit, mint, mutable, nn, ops
from mindspore.common import dtype as mstype
from vllm.attention.backends.abstract import AttentionMetadata, AttentionType
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.sampler import (SamplerOutput,
                                                get_sampler)

from vllm_mindspore.attention import Attention
from vllm_mindspore.model_executor.layers.rotary_embedding import get_rope
from vllm_mindspore.model_executor.models.attention_mask import (
    LowerTriangularMask)
from vllm_mindspore.model_executor.models.mindone_models.base import (
    MindONEModelBase)
from vllm_mindspore.model_executor.models.mindone_models.utils import (
    enable_dynamic_shape)
from vllm_mindspore.model_executor.models.model_base import Fake_Attention
from vllm_mindspore.model_executor.sampling_metadata import SamplingMetadata
from vllm_mindspore.utils import STR_DTYPE_TO_MS_DTYPE


class vLLMQwen3Attention(Qwen3Attention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.config.max_position_embeddings,
            base=self.config.rope_theta,
            rope_scaling=self.config.rope_scaling,
            dtype=mstype.bfloat16,
        )
        self.attn = Attention(
            self.config.num_attention_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.config.num_key_value_heads,
            prefix=f"model.layers.{self.layer_idx}.self_attn.attn",
            attn_type=AttentionType.DECODER)
        self.attn_mask = mint.triu(
            mint.ones(size=(128, 128), dtype=mstype.bfloat16), 1)
        self.hard_mask = Tensor([0], dtype=mstype.bfloat16).reshape(1, 1)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=self.config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=self.config.rms_norm_eps)

    @jit
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
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        #Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                           self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                           self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        q, k = self.rotary_emb(positions, q, k, batch_valid_length, is_prefill)
        attn_output = self.attn(q, k, v, key_cache, value_cache, is_prefill,
                                slot_mapping, attn_mask, batch_valid_length,
                                q_seq_lens, block_tables)
        output = self.o_proj(attn_output)
        return output


class vLLMQwen3DecoderLayer(nn.Cell):

    def __init__(self, config: "Qwen3Config", layer_idx: int):
        super().__init__()
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size,
                                            eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size,
                                                     eps=config.rms_norm_eps)

        self.self_attn = vLLMQwen3Attention(config, layer_idx)

    @jit
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
    ) -> Tuple[Tensor, Tensor]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(positions, hidden_states, key_cache,
                                       value_cache, is_prefill, slot_mapping,
                                       attn_mask, batch_valid_length,
                                       q_seq_lens, block_tables)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class vLLMQwen3Model(Qwen3PreTrainedModel):

    def __init__(self, config: "Qwen3Config"):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         padding_idx=self.padding_idx)
        self.layers = nn.CellList([
            vLLMQwen3DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

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
        block_tables: Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[Tensor] = None,
    ) -> Union[Tensor, IntermediateTensors]:

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        for i in range(len(self.layers)):

            hidden_states = self.layers[i](positions, hidden_states,
                                           key_caches[i], value_caches[i],
                                           is_prefill, slot_mapping, attn_mask,
                                           batch_valid_length, q_seq_lens,
                                           block_tables)

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, *args, **kwargs):
        pass


class vLLMQwen3ForCausalLM(Qwen3PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.model = vLLMQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Dense(config.hidden_size,
                                config.vocab_size,
                                has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()


class Qwen3ForCausalLM(MindONEModelBase):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # create model
        qwen3_mindone = vLLMQwen3ForCausalLM.from_pretrained(
            vllm_config.model_config.model, mindspore_dtype=mstype.bfloat16)
        self.model, self.lm_head = qwen3_mindone.model, qwen3_mindone.lm_head

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        self.sampler = get_sampler()

        self.set_modules({"model": self.model, "lm_head": self.lm_head})

        self.prefill = True
        self.mstype = STR_DTYPE_TO_MS_DTYPE.get(self.model_config.dtype,
                                                self.model_config.dtype)
        self.casual_mask = LowerTriangularMask(
            dtype=self.mstype, max_model_len=self.model_config.max_model_len)
        self.kv_caches = [
            Fake_Attention() for i in range(config.num_hidden_layers)
        ]
        compilation_config = vllm_config.compilation_config

        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(config.num_hidden_layers):
            compilation_config.static_forward_context[str(
                i)] = self.kv_caches[i]

    def get_input_embeddings(self, input_ids: Tensor) -> Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(self,
                input_ids: Tensor,
                positions: Tensor,
                kv_caches: List[Tuple[Tensor, Tensor]],
                attn_metadata: AttentionMetadata,
                intermediate_tensors: IntermediateTensors = None,
                inputs_embeds: Tensor = None,
                **kwargs) -> Union[Tensor, IntermediateTensors]:
        key_cache, value_cache = self.get_kvcache()

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
        is_prefill = bool(attn_metadata.num_decode_tokens == 0
                          and kv_cache_lens.max() == 0)
        if is_prefill:
            input_ids = ops.expand_dims(input_ids, 0)
        else:
            input_ids = ops.expand_dims(input_ids, 1)

        slot_mapping = attn_metadata.slot_mapping
        attn_mask = self.casual_mask.gen_attention_mask(
            is_prefill, positions, query_lens)
        seq_lens_np = np.array(attn_metadata.seq_lens, dtype=np.int32)
        batch_valid_length = Tensor.from_numpy(seq_lens_np)
        q_seq_lens = Tensor.from_numpy(
            np.array(attn_metadata.query_lens, dtype=np.int32))
        block_tables = attn_metadata.block_tables

        model_inputs = (\
            input_ids,
            positions,
            key_cache,
            value_cache,
            mutable(is_prefill),
            slot_mapping,
            attn_mask,
            batch_valid_length,
            q_seq_lens,
            block_tables,
            intermediate_tensors,
            inputs_embeds
        )

        if is_prefill:
            if not self.prefill:
                self.prefill = True
            enable_dynamic_shape(
                self.model, *model_inputs
            )  # enable dynamic shape once on first prefill step
        else:
            if self.prefill:
                self.prefill = False
                enable_dynamic_shape(
                    self.model, *model_inputs
                )  # enable dynamic shape once on first decode step

        model_output = self.model(*model_inputs)

        if is_prefill:
            model_output = ops.squeeze(model_output, 0)
        else:
            model_output = ops.squeeze(model_output, 1)

        return model_output

    def load_weights(self, *args, **kwargs):
        if self.config.tie_word_embeddings:
            self.lm_head.weight.set_data(
                self.model.embed_tokens.embedding_table.data)

    def sample(self, logits: Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def compute_logits(
        self,
        hidden_states: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[Tensor]:
        if sampling_metadata.selected_token_indices is not None:
            hidden_states = ops.gather(
                hidden_states, sampling_metadata.selected_token_indices, 0)

        logits = self.lm_head(hidden_states).float()

        return logits
