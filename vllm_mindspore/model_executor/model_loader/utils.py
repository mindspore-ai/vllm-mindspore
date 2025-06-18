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
import os

from mindspore import nn
from vllm.config import ModelConfig, ModelImpl
from vllm.model_executor.model_loader.utils import logger
from vllm.model_executor.models import ModelRegistry

from vllm_mindspore.model_executor.models.registry import (
    MindSporeModelRegistry, is_mf_mcore_archs)


def resolve_transformers_arch(model_config: ModelConfig,
                              architectures: list[str]):
    from mindone import transformers
    for i, arch in enumerate(architectures):
        if arch == "TransformersForCausalLM":
            continue
        auto_map: dict[str, str] = getattr(model_config.hf_config, "auto_map",
                                           None) or dict()
        if auto_map:
            logger(
                f"WARNING: loading model from remote_code is not support now,"
                f"but got {auto_map=}")

        model_module = getattr(transformers, arch, None)
        if model_module is None:
            raise ValueError(
                f"Cannot find model module. '{arch}' is not a registered "
                "model in the MindONE Transformers library.")

        # TODO(Isotr0py): Further clean up these raises.
        # perhaps handled them in _ModelRegistry._raise_for_unsupported?
        if model_config.model_impl == ModelImpl.TRANSFORMERS:
            if not model_module.is_backend_compatible():
                raise ValueError(
                    f"The Transformers implementation of {arch} is not "
                    "compatible with vLLM.")
            architectures[i] = "TransformersForCausalLM"
        if model_config.model_impl == ModelImpl.AUTO:
            if not model_module.is_backend_compatible():
                raise ValueError(
                    f"{arch} has no vLLM implementation and the Transformers "
                    "implementation is not compatible with vLLM. Try setting "
                    "VLLM_USE_V1=0.")
            logger.warning(
                "%s has no vLLM implementation, falling back to Transformers "
                "implementation. Some features may not be supported and "
                "performance may not be optimal.", arch)
            architectures[i] = "TransformersForCausalLM"
    return architectures


def get_ms_model_architecture(
        model_config: ModelConfig) -> tuple[type[nn.Cell], str]:
    architectures = getattr(model_config.hf_config, "architectures", [])
    if is_mf_mcore_archs(architectures):
        architectures.append("MindFormersForCausalLM")

    vllm_supported_archs = ModelRegistry.get_supported_archs()
    is_vllm_supported = any(arch in vllm_supported_archs
                            for arch in architectures)
    vllm_not_supported = not is_vllm_supported

    if os.getenv("vLLM_MODEL_BACKEND") == "MindONE" and (  # noqa: SIM112
            model_config.model_impl == ModelImpl.TRANSFORMERS or
            model_config.model_impl != ModelImpl.VLLM and vllm_not_supported):
        architectures = resolve_transformers_arch(model_config, architectures)
    elif vllm_not_supported:
        raise RuntimeError("vLLM-Mindspore does not support "
                           f"{str(architectures)} for now.")
    model_cls, arch = MindSporeModelRegistry.resolve_model_cls(architectures)
    if model_config.task == "embed":
        raise RecursionError("MindSpore unsupported embed model task now!")
    elif model_config.task == "classify":
        raise RecursionError("MindSpore unsupported classify model task now!")
    elif model_config.task == "reward":

        raise RecursionError("MindSpore unsupported reward model task now!")

    return model_cls, arch
