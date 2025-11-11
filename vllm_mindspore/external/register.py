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
"""register mindspore models"""

import os
from vllm import ModelRegistry
from vllm.logger import init_logger

logger = init_logger("vllm_mindspore.models")

def register_model():
    init_env()
    init_context()

    ModelRegistry.register_model(
        "Qwen2ForCausalLM",
        "vllm_mindspore.model_executor.models.qwen2:Qwen2ForCausalLM")

    ModelRegistry.register_model(
        "Qwen3ForCausalLM",
        "vllm_mindspore.model_executor.models.qwen3:Qwen3ForCausalLM")

    ModelRegistry.register_model(
        "Qwen3MoeForCausalLM",
        "vllm_mindspore.model_executor.models.qwen3_moe:Qwen3MoeForCausalLM")


def init_env():
    defact_env = {
        "MS_ENABLE_LCCL": "off",
        "HCCL_EXEC_TIMEOUT": "7200",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "MS_JIT_MODULES": "vllm_mindspore,research",
        "RAY_CGRAPH_get_timeout": "360",
        "MS_NODE_TIMEOUT": "180",
        "MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST": "FlashAttentionScore,PagedAttention"
    }

    for key, value in defact_env.items():
        if key not in os.environ:
            logger.debug('Setting %s to "%s"', key, value)
            os.environ[key] = value
    
def init_context():
    from mindspore import set_context
    set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
