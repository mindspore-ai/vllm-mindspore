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

from mindformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from mindformers.models.qwen3_moe.modeling_qwen3_moe import (  # noqa
    Qwen3MoeForCausalLM as Qwen3MoeForCausalLM_MF)
from mindspore.nn.utils import no_init_parameters
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_mindspore.model_executor.models.mf_models.config import (
    gen_model_config)
from vllm_mindspore.model_executor.models.mf_models.qwen3 import (
    Qwen3ForCausalLM)

logger = init_logger(__name__)


class Qwen3MoeForCausalLM(Qwen3ForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def _generate_model_config(self):
        self.mf_model_config = gen_model_config(self.mf_config, Qwen3MoeConfig)
        logger.debug("=====mf_model_config====\n%s", self.mf_model_config)

    def _create_network(self):
        # Initial network
        with no_init_parameters():  # Delay initialization
            network = Qwen3MoeForCausalLM_MF(self.mf_model_config)
        return network, network.model.output_layer
