#!/usr/bin/env python3
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
from abc import abstractmethod
from typing import Iterable, Optional, Set, Tuple, Union

import mindspore as ms
from mindformers.core.context import build_mf_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.tools.register.config import MindFormerConfig
from mindformers.tools.utils import is_pynative
from mindspore import Tensor, mint
from mindspore.common.api import _pynative_executor
from mindspore.communication import get_rank
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_dp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm_mindspore.model_executor.models.attention_mask import (
    LowerTriangularMask)
from vllm_mindspore.model_executor.models.model_base import MsModelBase

logger = init_logger(__name__)


class MfModelBase(MsModelBase):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        self.set_flags = False

        model_config_path = os.getenv("MINDFORMERS_MODEL_CONFIG")
        if model_config_path is None:
            raise RuntimeError(
                'For "MindFormers" model backend, environments MINDFORMERS_MODEL_CONFIG should be set!'
            )

        self.mf_config = MindFormerConfig(model_config_path)
        self.rank_id = get_rank()
        self.dp_size = get_dp_group()

        build_mf_context(self.mf_config)
        build_parallel_config(self.mf_config)
        self.mf_config.model.model_config.parallel_config = (
            self.mf_config.parallel_config)
        self.mf_config.model.model_config.parallel_config.model_parallel = (
            get_tensor_model_parallel_world_size())
        self.mf_config.model.model_config.parallel_config.pipeline_stage = 1
        self._generate_model_config()
        self.casual_mask = LowerTriangularMask(
            dtype=self.mf_model_config.compute_dtype,
            max_model_len=self.model_config.max_model_len)
        self.network, self.lm_head = self._create_network()

        affinity_config = self.mf_config.get('context',
                                             {}).get('affinity_cpu_list', {})
        if isinstance(affinity_config, dict):
            ms.runtime.set_cpu_affinity(True, affinity_config)

        self._set_dynamic_inputs()

    @abstractmethod
    def _generate_model_config(self):
        raise NotImplementedError(
            "Function _generate_model_config should be Implemented!")

    @abstractmethod
    def _create_network(self):
        raise NotImplementedError(
            "Function _create_network should be Implemented!")

    def _set_dynamic_inputs(self):
        self.network.set_dynamic_inputs()
        dynamic_hidden_states = Tensor(
            shape=[None, None], dtype=self.mf_model_config.compute_dtype)
        self.lm_head.set_inputs(dynamic_hidden_states)

    def prepare_inputs(self, input_ids, positions):
        return self.prepare_base_inputs(input_ids, positions)

    def update_model_inputs(self, model_inputs, **kwargs):
        return model_inputs

    def forward(self,
                input_ids: Tensor,
                positions: Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[Tensor] = None,
                **kwargs) -> Union[Tensor, IntermediateTensors]:
        model_inputs, is_prefill = self.prepare_inputs(input_ids, positions)
        model_inputs = self.update_model_inputs(model_inputs, **kwargs)

        # enable_mb_split is True in lager EP enable micro-batch and per-dp-bs > 1
        enable_mb_split = self.is_enable_micro_batch_split(
            is_prefill, model_inputs["q_seq_lens"])

        if is_prefill:
            if self.enable_micro_batch:
                self.network.phase = "prefill" if not enable_mb_split else "prefill_micro_batch"
                if not self.set_flags or is_pynative() or enable_mb_split:
                    self.network.add_flags_custom(is_first_iteration=True)
                    self.network.add_flags_enable_micro_batch(
                        enable_micro_batch=enable_mb_split)
            else:
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
            if selected_token_indices is not None and selected_token_indices.numel(
            ) <= 0:
                logits = ms.mint.zeros(
                    (0, self.mf_model_config.vocab_size),
                    dtype=self.mf_model_config.compute_dtype)
            else:
                hidden_states = hidden_states.index_select(
                    0, selected_token_indices)
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

    def is_enable_micro_batch_split(self, is_prefill, q_seq_lens):
        """Judge enable micro batch """
        if self.enable_micro_batch:
            is_prefill_cur_dp = mint.ones(
                (1), dtype=ms.int8) if is_prefill else mint.zeros(
                    (1), dtype=ms.int8)
            is_prefill_all_dp = get_dp_group().all_gather(is_prefill_cur_dp)
            return is_prefill_all_dp.sum(
            ) == self.dp_size and q_seq_lens.shape[0] > 1
        else:
            return False
