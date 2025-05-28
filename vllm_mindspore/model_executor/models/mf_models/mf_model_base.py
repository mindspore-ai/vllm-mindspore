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

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.distributed import get_tensor_model_parallel_world_size, get_pp_group
from vllm.logger import init_logger

import torch
import mindspore as ms
from mindspore import Tensor, mutable
from mindspore.common.api import _pynative_executor

from mindformers.tools.register.config import MindFormerConfig
from mindformers.core.context import build_mf_context
from mindformers.core.parallel_config import build_parallel_config

from vllm_mindspore.model_executor.models.model_base import MsModelBase

logger = init_logger(__name__)


def _pad_to_max(x, max_len):
    return x + [-1] * (max_len - len(x))


def _batch_seq(input_tokens, prefill):
    if prefill:
        return ms.ops.expand_dims(input_tokens, 0).to(ms.int32)

    return ms.mint.reshape(input_tokens, (-1, 1)).to(ms.int32)


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
        self.mf_config.model.model_config.parallel_config.pipeline_stage = get_pp_group().world_size

        self._generate_model_config()
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
        # dynamic_hidden_states = Tensor(shape=[None, None], dtype=self.mf_model_config.compute_dtype)
        # self.lm_head.set_inputs(dynamic_hidden_states)
        if get_pp_group().is_last_rank:
            dynamic_hidden_states = Tensor(shape=[None, None], dtype=self.mf_model_config.compute_dtype)
            self.lm_head.set_inputs(dynamic_hidden_states)
    
    def prepare_inputs(self, input_ids, positions, attn_metadata):
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
        if attn_metadata.num_decode_tokens == 0 and kv_cache_lens.max() == 0:
            is_prefill = True
        else:
            is_prefill = False

        q_seq_lens = ms.Tensor(query_lens_np, dtype=ms.int32)
        position_ids = ms.Tensor(positions, dtype=ms.int32)
        attention_mask = self.casual_mask.gen_attention_mask(is_prefill, position_ids, query_lens)

        model_inputs = {}
        model_inputs["input_ids"] = _batch_seq(input_ids, is_prefill)
        model_inputs["batch_valid_length"] = ms.Tensor.from_numpy(np.expand_dims(seq_lens_np, 0))
        model_inputs["block_tables"] = attn_metadata.block_tables
        model_inputs["slot_mapping"] = attn_metadata.slot_mapping
        model_inputs["position_ids"] = position_ids
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
        kv_caches: List[Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[Tensor] = None,
        **kwargs
    ) -> Union[Tensor, IntermediateTensors]:
        model_inputs, is_prefill = self.prepare_inputs(input_ids, positions, attn_metadata)
        model_inputs = self.update_model_inputs(model_inputs, **kwargs)
        model_inputs["hidden_states"] = intermediate_tensors["hidden_states"] if intermediate_tensors else None

        if is_prefill:
            self.network.phase = "prefill"
            if not self.set_flags:
                self.network.add_flags_custom(is_first_iteration=True)
            hidden_states = self.network(**model_inputs)
            self.network.phase = "increment"
            if not self.set_flags:
                self.network.add_flags_custom(is_first_iteration=False)
                self.set_flags = True
        else:
            hidden_states = self.network(**model_inputs)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
            })
        return hidden_states

    def compute_logits(
        self,
        hidden_states: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[Tensor]:
        selected_token_indices = sampling_metadata.selected_token_indices
        if selected_token_indices is not None and selected_token_indices.numel() <= 0:
            logits = ms.mint.zeros((0, self.mf_model_config.vocab_size),
                                    dtype=self.mf_model_config.compute_dtype)
        else:
            hidden_states = hidden_states.view(-1, self.mf_model_config.hidden_size)
            hidden_states = hidden_states.index_select(0, selected_token_indices)
            logits = self.lm_head(hidden_states)
            logits = logits.reshape(-1, logits.shape[-1])

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
