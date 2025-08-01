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

import sys

from vllm.model_executor.models.registry import (_LazyRegisteredModel,
                                                 _ModelRegistry, _run)

from vllm_mindspore.utils import (is_mindformers_model_backend,
                                  is_mindone_model_backend)

_NATIVE_MODELS = {
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
}

_MINDFORMERS_MODELS = {
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM"),  # MCore
    "Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),  # MCore
    "DeepseekV3ForCausalLM": ("deepseek_v3", "DeepseekV3ForCausalLM"),
    "DeepSeekMTPModel": ("deepseek_mtp", "DeepseekV3MTPForCausalLM"),
}

_MINDONE_MODELS = {
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen2_5_VLForConditionalGeneration":
    ("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration"),
    "Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM"),
}

_registry_dict = {}
if is_mindformers_model_backend():
    _registry_dict = {
        model_arch: _LazyRegisteredModel(
            module_name=f"vllm_mindspore.model_executor.models.mf_models.{mod_relname}",
            class_name=cls_name,
        )
        for model_arch, (mod_relname, cls_name) in _MINDFORMERS_MODELS.items()
    }
elif is_mindone_model_backend():
    _registry_dict = {
        model_arch: _LazyRegisteredModel(
            module_name=
            f"vllm_mindspore.model_executor.models.mindone_models.{mod_relname}",
            class_name=cls_name,
        )
        for model_arch, (mod_relname, cls_name) in _MINDONE_MODELS.items()
    }
else:
    _registry_dict = {
        model_arch: _LazyRegisteredModel(
            module_name=f"vllm_mindspore.model_executor.models.{mod_relname}",
            class_name=cls_name,
        )
        for model_arch, (mod_relname, cls_name) in _NATIVE_MODELS.items()
    }

MindSporeModelRegistry = _ModelRegistry(_registry_dict)

_SUBPROCESS_COMMAND = [
    sys.executable, "-m", "vllm_mindspore.model_executor.models.registry"
]

if __name__ == "__main__":
    _run()
