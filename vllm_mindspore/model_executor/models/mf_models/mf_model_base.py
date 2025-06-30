#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
from types import MethodType
from typing import Iterable, List, Optional, Set, Tuple, Union
from abc import abstractmethod
import numpy as np
import math

from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.forward_context import get_forward_context
import vllm.envs as envs

import mindspore as ms
from mindspore import Tensor
from mindspore.common.api import _pynative_executor

from mindformers.tools.register.config import MindFormerConfig
from mindformers.core.context import build_mf_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.tools.utils import is_pynative

from vllm_mindspore.model_executor.models.model_base import MsModelBase
from vllm_mindspore.model_executor.models.attention_mask import LowerTriangularMask
from vllm_mindspore.v1.attention.backends.flash_attn import FlashAttentionMetadata

logger = init_logger(__name__)

class MfModelBase(MsModelBase):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super(MfModelBase, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )

        self.mf_config = MindFormerConfig(os.getenv("MINDFORMERS_MODEL_CONFIG"))
        build_mf_context(self.mf_config)
        build_parallel_config(self.mf_config)
        self.mf_config.model.model_config.parallel_config = (
            self.mf_config.parallel_config
        )
        self.mf_config.model.model_config.parallel_config.model_parallel = (
            get_tensor_model_parallel_world_size()
        )
        self.mf_config.model.model_config.parallel_config.pipeline_stage = 1
        self._generate_model_config()
        self.casual_mask = LowerTriangularMask(dtype=self.mf_model_config.compute_dtype,
                                               max_model_len=self.mf_model_config.seq_length)
        self.network, self.lm_head = self._create_network()

        affinity_config = self.mf_config.get('context', {}).get('affinity_cpu_list', {})
        if isinstance(affinity_config, dict):
            ms.runtime.set_cpu_affinity(True, affinity_config)

        self._set_dynamic_inputs()

    @abstractmethod
    def _generate_model_config(self):
        raise NotImplementedError("Function _generate_model_config should be Implemented!")

    @abstractmethod
    def _create_network(self):
        raise NotImplementedError("Function _create_network should be Implemented!")

    def _set_dynamic_inputs(self):
        self.network.set_dynamic_inputs()
        dynamic_hidden_states = Tensor(shape=[None, None], dtype=self.mf_model_config.compute_dtype)
        self.lm_head.set_inputs(dynamic_hidden_states)

    def _dummy_attention_metadata(self, input_ids: Tensor, positions: Tensor) -> FlashAttentionMetadata:
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

    def prepare_inputs(self, input_ids, positions, attn_metadata):
        key_cache, value_cache = self.get_kvcache()
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
            if attn_metadata.num_decode_tokens == 0 and kv_cache_lens.max() == 0:
                is_prefill = True
            else:
                is_prefill = False

            q_seq_lens = ms.Tensor(query_lens_np, dtype=ms.int32)
            position_ids = ms.Tensor(positions, dtype=ms.int32)
            attention_mask = self.casual_mask.gen_attention_mask(is_prefill, position_ids, query_lens)

            model_inputs = {}
            model_inputs["input_ids"] = input_ids.astype(ms.int32)
            model_inputs["batch_valid_length"] = ms.from_numpy(seq_lens_np)
            model_inputs["block_tables"] = attn_metadata.block_tables
            model_inputs["slot_mapping"] = attn_metadata.slot_mapping
            model_inputs["position_ids"] = position_ids
            model_inputs["q_seq_lens"] = q_seq_lens
            model_inputs["attention_mask"] = attention_mask
            model_inputs["key_cache"] = key_cache
            model_inputs["value_cache"] = value_cache
        else:
            if attn_metadata.max_context_lens == 0:
                is_prefill = True
            else:
                is_prefill = False
            q_seq_lens = attn_metadata.q_seq_lens
            query_lens_np = attn_metadata.q_seq_lens_np
            attention_mask = self.casual_mask.gen_attention_mask(is_prefill, positions, query_lens_np)

            model_inputs = {}
            model_inputs["input_ids"] = input_ids.astype(ms.int32)
            model_inputs["batch_valid_length"] = ms.from_numpy(attn_metadata.seq_lens_np)
            model_inputs["block_tables"] = attn_metadata.block_tables
            model_inputs["slot_mapping"] = attn_metadata.slot_mapping
            model_inputs["position_ids"] = positions.to(ms.int32)
            model_inputs["q_seq_lens"] = q_seq_lens
            model_inputs["attention_mask"] = attention_mask
            model_inputs["key_cache"] = key_cache
            model_inputs["value_cache"] = value_cache

        return model_inputs, is_prefill

    def update_model_inputs(self, model_inputs, **kwargs):
        return model_inputs

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[Tensor] = None,
        **kwargs
    ) -> Union[Tensor, IntermediateTensors]:
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            attn_metadata = self._dummy_attention_metadata(input_ids, positions)
        model_inputs, is_prefill = self.prepare_inputs(input_ids, positions, attn_metadata)
        model_inputs = self.update_model_inputs(model_inputs, **kwargs)

        if is_prefill:
            self.network.phase = "prefill"
            if not self.set_flags or is_pynative():
                self.network.add_flags_custom(is_first_iteration=True)
            hidden_states = self.network(**model_inputs)
            self.network.phase = "increment"
            if not self.set_flags or is_pynative():
                self.network.add_flags_custom(is_first_iteration=False)
                self.set_flags = True
        else:
            hidden_states = self.network(**model_inputs)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[Tensor]:
        if sampling_metadata is not None:
            selected_token_indices = sampling_metadata.selected_token_indices
            if selected_token_indices is not None and selected_token_indices.numel() <= 0:
                logits = ms.mint.zeros((0, self.mf_model_config.vocab_size),
                                        dtype=self.mf_model_config.compute_dtype)
            else:
                hidden_states = hidden_states.index_select(0, selected_token_indices)
                logits = self.lm_head(hidden_states)
                logits = logits.view(-1, logits.shape[-1])
        else:
            logits = self.lm_head(hidden_states)
            logits = logits.view(-1, logits.shape[-1])
        return logits

    def sample(
        self,
        logits: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        _pynative_executor.sync()
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> Set[str]:
        raise NotImplementedError("load_weight not implemented.")
