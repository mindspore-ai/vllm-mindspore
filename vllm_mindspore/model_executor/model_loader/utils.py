# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/model_executor/model_loader/utils.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2024-2025 The vLLM team.
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

""" utils for load model """
from mindspore import nn
from vllm.config import ModelConfig
from vllm.model_executor.models import ModelRegistry

from vllm_mindspore.model_executor.models.registry import (
    is_mf_mcore_archs, _MINDFORMERS_MODELS)
from vllm_mindspore.utils import (is_mindformers_model_backend,
                                  is_mindone_model_backend)


def get_ms_model_architecture(
        model_config: ModelConfig) -> tuple[type[nn.Cell], str]:
    architectures = getattr(model_config.hf_config, "architectures", [])

    if is_mindformers_model_backend():
        is_mf_norm_model = any(arch in _MINDFORMERS_MODELS for arch in architectures)
        if not is_mf_norm_model:
            architectures.clear()
            architectures.append("MindFormersForCausalLM")
    elif is_mindone_model_backend():
        raise RuntimeError("MindOne Models to be supported.")

    vllm_supported_archs = ModelRegistry.get_supported_archs()
    is_vllm_supported = any(arch in vllm_supported_archs
                            for arch in architectures)
    if not is_vllm_supported:
        raise RuntimeError("vLLM-Mindspore does not support "
                           f"{str(architectures)} for now.")
    # model_cls, arch = MindSporeModelRegistry.resolve_model_cls(architectures)
    model_cls, arch = ModelRegistry.resolve_model_cls(architectures)
    if model_config.task == "embed":
        raise RecursionError("MindSpore unsupported embed model task now!")
    elif model_config.task == "classify":
        raise RecursionError("MindSpore unsupported classify model task now!")
    elif model_config.task == "reward":

        raise RecursionError("MindSpore unsupported reward model task now!")

    return model_cls, arch
