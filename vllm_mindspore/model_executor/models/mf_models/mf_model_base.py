# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologies Co., Ltd.
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

import os
from abc import abstractmethod
from collections.abc import Iterable
from typing import Optional, Union

import mindspore as ms
from mindformers.core.context import build_mf_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.tools.register.config import MindFormerConfig
from mindformers.tools.utils import is_pynative
from mindspore import Tensor, nn
from mindspore.common.api import _pynative_executor
from mindspore.communication import get_rank
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm_mindspore.model_executor.models.attention_mask import (
    LowerTriangularMask)
from vllm_mindspore.model_executor.models.model_base import MsModelBase

try:
    # Need to apply dllm pd patch on vllm to use pd disagg related functions
    from vllm.attention.layer import (maybe_save_kv_layer_to_connector,
                                      wait_for_kv_layer_from_connector)
    from vllm.distributed.kv_transfer import is_v1_kv_transfer_group
    kv_transfer_supported = True
except:  # noqa: E722
    kv_transfer_supported = False

logger = init_logger(__name__)


class MfModelBase(MsModelBase):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        model_config_path = os.getenv("MINDFORMERS_MODEL_CONFIG")
        if model_config_path is None:
            raise RuntimeError('For "MindFormers" model backend, environments '
                               'MINDFORMERS_MODEL_CONFIG should be set!')

        self.mf_config = MindFormerConfig(model_config_path)
        self.rank_id = get_rank()
        self.dp_size = get_dp_group()

        self.kv_transfer_config = vllm_config.kv_transfer_config
        build_mf_context(self.mf_config)
        build_parallel_config(self.mf_config)
        self.mf_config.model.model_config.parallel_config = (
            self.mf_config.parallel_config)
        self.mf_config.model.model_config.parallel_config.model_parallel = (
            get_tensor_model_parallel_world_size())
        self.mf_config.model.model_config.parallel_config.pipeline_stage = 1
        self.use_mla_op = \
            bool(vllm_config.additional_config
                 and vllm_config.additional_config.get('use_mla_op') == 1)
        self._generate_model_config()
        if not hasattr(self, 'mf_model_config'):
            raise RuntimeError('mf_model_config not initialized')
        self.casual_mask = LowerTriangularMask(
            dtype=self.mf_model_config.compute_dtype,
            max_model_len=self.model_config.max_model_len)
        self.network, self.lm_head = self._create_network()

        affinity_config = self.mf_config.get('context',
                                             {}).get('affinity_cpu_list', {})
        if isinstance(affinity_config, dict):
            ms.runtime.set_cpu_affinity(True, affinity_config)

        self._set_dynamic_inputs()

    @property
    def ready_lm_head(self) -> nn.Cell:
        if self.lm_head is None:
            raise RuntimeError("lm_head not initialized")
        return self.lm_head

    @abstractmethod
    def _generate_model_config(self):
        raise NotImplementedError(
            "Function _generate_model_config should be Implemented!")

    @abstractmethod
    def _create_network(self):
        raise NotImplementedError(
            "Function _create_network should be Implemented!")

    # DLLM
    def is_decoder_task(self) -> bool:
        if self.kv_transfer_config is None:
            return False

        return self.kv_transfer_config.is_kv_consumer

    # DLLM
    def is_prefill_task(self) -> bool:
        if self.kv_transfer_config is None:
            return False

        return self.kv_transfer_config.is_kv_producer

    def _set_dynamic_inputs(self):
        self.network.set_dynamic_inputs()
        if not hasattr(self, 'mf_model_config'):
            raise RuntimeError('mf_model_config not initialized')
        dynamic_hidden_states = Tensor(
            shape=[None, None], dtype=self.mf_model_config.compute_dtype)
        self.ready_lm_head.set_inputs(dynamic_hidden_states)

    def prepare_inputs(self, input_ids, positions):
        return self.prepare_base_inputs(input_ids, positions)

    def update_model_inputs(self, model_inputs, **kwargs):
        return model_inputs

    # DLLM
    def connector_send_kvcache(self):
        logger.debug("reached connector_send_kvcache")
        _pynative_executor.sync()
        forward_context = get_forward_context()
        if not hasattr(self, 'mf_model_config'):
            raise RuntimeError('mf_model_config not initialized')
        for i in range(self.mf_model_config.num_layers):
            kv_cache = self.kv_caches[i]
            k_cache = kv_cache.kv_cache[forward_context.virtual_engine][0]
            v_cache = kv_cache.kv_cache[forward_context.virtual_engine][1]
            maybe_save_kv_layer_to_connector(str(i), (k_cache, v_cache))

    # DLLM
    def connector_wait_for_kv_layer(self):
        logger.debug("reached connector_wait_for_kv_layer")
        if not hasattr(self, 'mf_model_config'):
            raise RuntimeError('mf_model_config not initialized')
        for i in range(self.mf_model_config.num_layers):
            wait_for_kv_layer_from_connector("key." + str(i))

    def forward(self,
                input_ids: Tensor,
                positions: Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[Tensor] = None,
                **kwargs) -> Union[Tensor, IntermediateTensors]:
        model_inputs, is_prefill = self.prepare_inputs(input_ids, positions)
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
            if kv_transfer_supported and is_v1_kv_transfer_group():
                self.connector_send_kvcache()
        # DLLM
        else:
            if kv_transfer_supported:
                if is_v1_kv_transfer_group() and self.is_prefill_task():
                    self.connector_send_kvcache()

                if is_v1_kv_transfer_group() and self.is_decoder_task():
                    self.connector_wait_for_kv_layer()
                    logger.debug("connector_wait_for_kv_layer success")
            hidden_states = self.network(**model_inputs)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[Tensor]:
        if sampling_metadata is not None:
            selected_token_indices = sampling_metadata.selected_token_indices
            if (selected_token_indices is not None
                    and selected_token_indices.numel() <= 0):
                if not hasattr(self, 'mf_model_config'):
                    raise RuntimeError('mf_model_config not initialized')
                logits = ms.mint.zeros(
                    (0, self.mf_model_config.vocab_size),
                    dtype=self.mf_model_config.compute_dtype)
            else:
                hidden_states = hidden_states.index_select(
                    0, selected_token_indices)
                logits = self.ready_lm_head(hidden_states)
                logits = logits.view(-1, logits.shape[-1])
        else:
            logits = self.ready_lm_head(hidden_states)
            logits = logits.view(-1, logits.shape[-1])
        return logits

    def sample(
        self,
        logits: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        if not hasattr(self, 'sampler'):
            raise RuntimeError('sampler not initialized')
        next_tokens = self.sampler(logits, sampling_metadata)
        _pynative_executor.sync()
        return next_tokens

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        raise NotImplementedError("load_weight not implemented.")
