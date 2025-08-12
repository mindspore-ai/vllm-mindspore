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

def register_models():
    from vllm import ModelRegistry

    # native model
    ModelRegistry.register_model(
        "Qwen3ForCausalLM",
        "vllm_mindspore.model_executor.models.qwen3:Qwen3ForCausalLM")

    ModelRegistry.register_model(
        "Qwen2ForCausalLM",
        "vllm_mindspore.model_executor.models.qwen2:Qwen2ForCausalLM")

    ModelRegistry.register_model(
        "Qwen2_5_VLForConditionalGeneration",
        "vllm_mindspore.model_executor.models.qwen2_5_vl:Qwen2_5_VLForConditionalGeneration")

    ModelRegistry.register_model(
        "LlamaForCausalLM",
        "vllm_mindspore.model_executor.models.llama:LlamaForCausalLM")

    # mindformers model
    ModelRegistry.register_model(
        "DeepSeekMTPModel",
        "vllm_mindspore.model_executor.models.mf_models.deepseek_v3:DeepseekV3ForCausalLM")

    ModelRegistry.register_model(
        "DeepseekV3MTPForCausalLM",
        "vllm_mindspore.model_executor.models.mf_models.deepseek_mtp:DeepseekV3MTPForCausalLM")

    ModelRegistry.register_model(
        "MindFormersForCausalLM",
        "vllm_mindspore.model_executor.models.mf_models.mindformers:MindFormersForCausalLM")

    # mindone model

    from .model_processor import apply_model_patch
    apply_model_patch()
