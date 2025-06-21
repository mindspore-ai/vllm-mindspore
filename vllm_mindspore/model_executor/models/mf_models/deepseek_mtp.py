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

from typing import Iterable, Set, Tuple

from vllm.config import VllmConfig
from vllm.config import  get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from mindspore import Tensor, JitConfig, Model, mutable
from mindspore.nn.utils import no_init_parameters

from research.deepseek3.deepseek3_config import (
    DeepseekV3Config as DeepseekV3Config_MF,
)
from research.deepseek3.deepseek3 import (
    DeepseekV3ForCausalLM as DeepseekV3ForCausalLM_MF,
)

from vllm_mindspore.model_executor.layers.sampler import get_sampler
from vllm_mindspore.model_executor.models.model_base import MLAAttentionWrapper
from vllm_mindspore.model_executor.models.mf_models.mf_model_base import MfModelBase
from vllm_mindspore.model_executor.models.mf_models.deepseekv3_weight_processor import DeepseekV3WeightProcessor

logger = init_logger(__name__)

class DeepseekV3MTPForCausalLM(MfModelBase):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super(DeepseekV3MTPForCausalLM, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )
        self.mf_kvcaches_init = False

        self.sampler = get_sampler()
        self.set_modules({"model": self.network})

        self.kv_caches = [MLAAttentionWrapper() for i in range(self.mf_model_config.num_layers)]
        compilation_config = get_current_vllm_config().compilation_config

        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(self.mf_model_config.num_nextn_predict_layers):
            compilation_config.static_forward_context[str(i)] = self.kv_caches[i]

        self.set_flags = False


    def _generate_model_config(self):
        self.mf_config.load_checkpoint = self.get_model_path()

        self.mf_model_config = DeepseekV3Config_MF(**self.mf_config.model.model_config)
        if self.mf_config.moe_config:
            self.mf_model_config.moe_config = self.mf_config.moe_config
        self.mf_model_config.return_hidden_states = True
        setattr(self.mf_model_config, 'npu_mem_size', -1)

        self.mf_model_config.is_mtp_model = True
        self.mf_model_config.num_nextn_predict_layers = self.model_config.hf_config.num_nextn_predict_layers
        if self.mf_model_config.num_nextn_predict_layers != 1:
            raise NotImplementedError("Only support 1 MTP-layer now.")

        self.mf_config.model.model_config = self.mf_model_config


    def _create_network(self):
        # Initital network
        with no_init_parameters():  # Delay initialization
            network = DeepseekV3ForCausalLM_MF(self.mf_model_config)

        return network, network.mtp_model.head


    def get_kvcache(self):
        key_cache = []
        forward_context = get_forward_context()
        for i in range(self.mf_model_config.num_nextn_predict_layers):
            k_cache = self.kv_caches[i].kv_cache[forward_context.virtual_engine][0]
            key_cache.append(k_cache)
        return mutable(key_cache), None


    def update_model_inputs(self, model_inputs, **kwargs):
        # ToDo: supports multi-mtpLayers with 'spec_step_idx' specifing the layer index.
        if kwargs.get("spec_step_idx", 0) != 0:
            raise NotImplementedError("Only support 1 MTP-layer now.")
        # model_inputs["index"] = ms.Tensor(kwargs.get("spec_step_idx", 0), ms.int32)
        hidden_states_shape = list(model_inputs["input_ids"].shape)
        hidden_states_shape.append(self.model_config.get_hidden_size())
        model_inputs["hidden_states"] = kwargs.get("previous_hidden_states").reshape(hidden_states_shape)
        return model_inputs


    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> Set[str]:
        weight_processor = DeepseekV3WeightProcessor(self.mf_config, self.network, False)
        weight_processor.load_safetensors_shard(self.mf_config.load_checkpoint, is_mtp_model=True)
        self.network.set_dynamic_inputs()
        dynamic_hidden_states = Tensor(shape=[None, None], dtype=self.mf_model_config.compute_dtype)
        self.lm_head.set_inputs(dynamic_hidden_states)
        return None
