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

from typing import Iterable, Optional, Tuple, Union

import mindspore as ms
import numpy as np
from mindformers.core.context import build_mf_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.models.qwen3.configuration_qwen3 import Qwen3Config
from mindformers.models.qwen3.modeling_qwen3 import (  # noqa
    Qwen3ForCausalLM as Qwen3ForCausalLM_MF)
from mindformers.tools.utils import is_pynative
from mindspore import Tensor, ops
from mindspore.common.api import _pynative_executor
from mindspore.nn.utils import no_init_parameters
from vllm import envs
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm_mindspore.model_executor.layers.sampler import get_sampler
from vllm_mindspore.model_executor.models.attention_mask import (
    LowerTriangularMask)
from vllm_mindspore.model_executor.models.mf_models.config import (
    gen_mf_config, gen_model_config)
from vllm_mindspore.model_executor.models.model_base import (AttentionWrapper,
                                                             MsModelBase)

logger = init_logger(__name__)


class Qwen3ForCausalLM(MsModelBase):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.set_flags = False

        mf_config = gen_mf_config(vllm_config)
        mf_config.load_checkpoint = self.get_model_path()
        self.mf_config = mf_config

        build_mf_context(self.mf_config)
        build_parallel_config(self.mf_config)

        self._generate_model_config()
        self.casual_mask = LowerTriangularMask(
            dtype=self.mf_model_config.compute_dtype,
            max_model_len=self.mf_model_config.seq_length)
        self.network, self.lm_head = self._create_network()

        affinity_config = self.mf_config.get('context',
                                             {}).get('affinity_cpu_list', {})
        if isinstance(affinity_config, dict):
            ms.runtime.set_cpu_affinity(True, affinity_config)

        self._set_dynamic_inputs()

        self.sampler = get_sampler()
        self.set_modules({"model": self.network})
        self.kv_caches = [
            AttentionWrapper()
            for _ in range(self.mf_model_config.num_hidden_layers)
        ]
        compilation_config = get_current_vllm_config().compilation_config

        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(self.mf_model_config.num_hidden_layers):
            compilation_config.static_forward_context[str(
                i)] = self.kv_caches[i]

        self.cast = ops.Cast()

    def _set_dynamic_inputs(self):
        self.network.set_dynamic_inputs()
        dynamic_hidden_states = Tensor(
            shape=[None, None], dtype=self.mf_model_config.compute_dtype)
        self.lm_head.set_inputs(dynamic_hidden_states)

    def prepare_inputs(self, input_ids, positions):

        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            attn_metadata = self._dummy_attention_metadata(
                input_ids, positions)
        key_cache, value_cache = self.get_kvcache()
        if not envs.VLLM_USE_V1:
            # V0
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
            if attn_metadata.num_decode_tokens == 0 and kv_cache_lens.max(
            ) == 0:
                is_prefill = True
            else:
                is_prefill = False
            context_lens_tensor = ms.from_numpy(kv_cache_lens)
        else:
            # V1
            is_prefill = attn_metadata.max_context_lens == 0
            query_lens_np = attn_metadata.q_seq_lens_np
            seq_lens_np = attn_metadata.seq_lens_np
            context_lens_tensor = attn_metadata.context_lens

        q_seq_lens = ms.Tensor(query_lens_np, dtype=ms.int32)
        position_ids = ms.Tensor(positions, dtype=ms.int32)
        attention_mask = self.casual_mask.gen_attention_mask(
            is_prefill, positions, query_lens_np)

        model_inputs = {}
        model_inputs["input_ids"] = input_ids.astype(ms.int32)
        model_inputs["batch_valid_length"] = ms.from_numpy(seq_lens_np)
        model_inputs["block_tables"] = attn_metadata.block_tables
        model_inputs["slot_mapping"] = attn_metadata.slot_mapping
        model_inputs["positions"] = position_ids
        model_inputs["q_seq_lens"] = q_seq_lens
        model_inputs["attention_mask"] = attention_mask
        model_inputs["key_cache"] = key_cache
        model_inputs["value_cache"] = value_cache
        model_inputs["context_lens_tensor"] = context_lens_tensor

        return model_inputs, is_prefill

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
                self.network.add_flags_custom_mcore(is_prefill=True)
            hidden_states = self.network(**model_inputs)
            self.network.phase = "increment"
            if not self.set_flags or is_pynative():
                self.network.add_flags_custom_mcore(is_prefill=False)
                self.set_flags = True
        else:
            hidden_states = self.network(**model_inputs)

        return hidden_states

    def _generate_model_config(self):
        self.mf_model_config = gen_model_config(self.mf_config, Qwen3Config)
        logger.debug("=====mf_model_config====\n", self.mf_model_config)

    def _create_network(self):
        # Initial network
        with no_init_parameters():  # Delay initialization
            network = Qwen3ForCausalLM_MF(self.mf_model_config)
        return network, network.model.output_layer

    def update_model_inputs(self, model_inputs, **kwargs):
        return model_inputs

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
                hidden_states = hidden_states.reshape(
                    (-1, hidden_states.shape[-1]))
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

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]):
        self.network.load_weights(self.mf_config.load_checkpoint)
        self.network.set_dynamic_inputs()
        return None
