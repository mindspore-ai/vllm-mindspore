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
    AUTO_SELECT_FIXED_MODEL, MindSporeModelRegistry, mcore_support_list,
    mf_supported, mindone_supported)
from vllm_mindspore.utils import (is_mindformers_model_backend,
                                  is_mindone_model_backend,
                                  is_mix_model_backend)


def mf_mcore_compatible(arch):
    # vllm overrides the model arch to `DeepSeekMTPModel` for mtp model,
    # which is not registered in mf independently and is
    # a sub-class of `DeepseekV3ForCausalLM`.
    return arch in mcore_support_list or arch == "DeepSeekMTPModel"


def resolve_mf_mcore_arch(model_config: ModelConfig, architectures: list[str]):
    for i, arch in enumerate(architectures):
        if arch == "TransformersForCausalLM":
            architectures[i] = "MindFormersForCausalLM"
            continue
        if mf_mcore_compatible(arch):
            architectures[i] = "MindFormersForCausalLM"
        else:
            raise ValueError(
                f'"{arch}" is not supported in MindFormers Backend. '
                f'Supported architectures: {mcore_support_list}')


def resolve_mindone_transformers_arch(model_config: ModelConfig,
                                      architectures: list[str]):
    from mindone import transformers
    for i, arch in enumerate(architectures):
        if arch == "TransformersForCausalLM":
            continue
        auto_map: dict[str, str] = getattr(model_config.hf_config, "auto_map",
                                           None) or dict()
        if auto_map:
            logger.warning(
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


def resolve_mix_backend_architecture(model_config, architectures):
    """
    Resolve the backend architecture for the given model according 
    to the following priority strategy:

    1. If the architecture is in the list of AUTO_SELECT_FIXED_MODEL,
       skip it (do not modify).
    2. MindFormers backend has the highest priority. If MindFormers is 
       supported and the architecture is mcore compatible, override the 
       architecture to "MindFormersForCausalLM".
    3. If the architecture has been supported by the Native or Mindformers
       backend, keep it as is.
    4. If MindOne Transformers backend is supported and the architecture exists
       in transformers, override the architecture to "TransformersForCausalLM".

    This strategy ensures that the most optimal and compatible backend 
    is selected for each architecture, prioritizing MindFormers, then vLLM,
    and finally MindOne Transformers as a fallback.
    """
    if mindone_supported:
        from mindone import transformers
    vllm_supported_archs = ModelRegistry.get_supported_archs()
    models = ModelRegistry.models
    for i, arch in enumerate(architectures):
        if arch in AUTO_SELECT_FIXED_MODEL:
            m = AUTO_SELECT_FIXED_MODEL[arch]
            m_path = m[0]
            logger.info('arch "%s" is run with fixed path "%s"', arch, m_path)
            continue
        # MindFormers backend has the highest priority:
        # If mcore model supported the architectures and
        # MindFormers backend does not directly support this model,
        # Use mcore model.as backend.
        if mf_supported and mf_mcore_compatible(arch) and \
           (arch not in vllm_supported_archs or \
            arch in vllm_supported_archs and
            "mf_models" not in models[arch].module_name):

            architectures[i] = "MindFormersForCausalLM"
            logger.info('arch "%s" set to "MindFormersForCausalLM"', arch)

        # If MindFormers or Native backend can support, keep it.
        if architectures[i] in vllm_supported_archs:
            m = models[architectures[i]]
            if "mf_model" in m.module_name:
                logger.info('"%s" run as "%s" with MindFormers backend.', arch,
                            architectures[i])
                if os.getenv("MINDFORMERS_MODEL_CONFIG", None):
                    raise ValueError(
                        "MINDFORMERS_MODEL_CONFIG is not supported, "
                        "please unset MINDFORMERS_MODEL_CONFIG. "
                        "For usage of vllm_mindspore, refer to "
                        "https://www.mindspore.cn/vllm_mindspore/"
                        "docs/zh-CN/master/getting_started/"
                        "quick_start/quick_start.html")
            else:
                logger.info('"%s" run as "%s" with Native backend.', arch,
                            architectures[i])
            continue

        # Try MindOne Transformers backend if supported and available.
        if mindone_supported and getattr(transformers, arch, None) is not None:
            architectures[i] = "TransformersForCausalLM"
            logger.info('arch "%s" set to MindONE "TransformersForCausalLM"',
                        arch)

        if architectures[i] in vllm_supported_archs:
            logger.info('arch "%s" run as "%s" with MindONE backend.', arch,
                        architectures[i])
        else:
            raise ValueError(f'The model "{arch}" is not compatible '
                             'with vLLM-MindSpore Plugin.')


def is_vllm_supported(architectures):
    vllm_supported_archs = ModelRegistry.get_supported_archs()
    return any(arch in vllm_supported_archs for arch in architectures)


def get_ms_model_architecture(
        model_config: ModelConfig) -> tuple[type[nn.Cell], str]:
    architectures = getattr(model_config.hf_config, "architectures", [])

    # Select the architecture resolution strategy based on backend settings.
    # For native backend, no resolution is needed.
    if is_mix_model_backend():
        resolve_mix_backend_architecture(model_config, architectures)
    elif is_mindformers_model_backend():
        if not is_vllm_supported(architectures):  # noqa: SIM102
            resolve_mf_mcore_arch(model_config, architectures)
    elif is_mindone_model_backend():  # noqa: SIM102
        if not is_vllm_supported(architectures):
            resolve_mindone_transformers_arch(model_config, architectures)

    model_cls, arch = MindSporeModelRegistry.resolve_model_cls(
        architectures, model_config)
    if model_config.task == "embed":
        raise RecursionError("MindSpore unsupported embed model task now!")
    elif model_config.task == "classify":
        raise RecursionError("MindSpore unsupported classify model task now!")
    elif model_config.task == "reward":
        raise RecursionError("MindSpore unsupported reward model task now!")

    return model_cls, arch
