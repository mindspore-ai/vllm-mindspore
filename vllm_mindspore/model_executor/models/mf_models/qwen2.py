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

from collections.abc import Iterable

from mindformers.models.llama import LlamaConfig as LlamaConfig_MF
from mindspore import Tensor
from mindspore.nn.utils import no_init_parameters
# yapf conflict with isort
# yapf: disable  # noqa: ERA001
from research.qwen2_5.infer.qwen2_5 import (
    ParallelQwenForCausalLM as ParallelQwenForCausalLM_MF)
# yapf: enable  # noqa: ERA001
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import get_sampler

from vllm_mindspore.model_executor.models.mf_models.mf_model_base import (
    MfModelBase)
from vllm_mindspore.model_executor.models.mf_models.qwen2_weight_processor import (  # noqa: E501
    Qwen2WeightProcessor)
from vllm_mindspore.model_executor.models.model_base import AttentionWrapper

logger = init_logger(__name__)


class Qwen2ForCausalLM(MfModelBase):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.mf_kvcaches_init = False

        self.sampler = get_sampler()
        self.set_modules({"model": self.network})

        self.kv_caches = [
            AttentionWrapper() for i in range(self.mf_model_config.num_layers)
        ]
        compilation_config = get_current_vllm_config().compilation_config

        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(self.mf_model_config.num_layers):
            compilation_config.static_forward_context[str(
                i)] = self.kv_caches[i]

        self.set_flags = False

    def _generate_model_config(self):
        self.mf_config.load_checkpoint = self.get_model_path()
        self.mf_model_config = LlamaConfig_MF(
            **self.mf_config.model.model_config)
        if self.mf_config.moe_config:
            self.mf_model_config.moe_config = self.mf_config.moe_config
        self.mf_model_config.return_hidden_states = True

        # qwen qkv concat will support in next version
        self.mf_model_config.qkv_concat = False
        self.mf_model_config.npu_mem_size = -1
        self.mf_config.model.model_config.qkv_concat = False

    def _create_network(self):
        # Initial network
        with no_init_parameters():  # Delay initialization
            network = ParallelQwenForCausalLM_MF(self.mf_model_config)
        return network, network.lm_head

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        weight_processor = Qwen2WeightProcessor(self.mf_config, self.network,
                                                False, weights)
        weight_processor.load_safetensors_shard(self.mf_config.load_checkpoint)

        return None
