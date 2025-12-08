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
from typing import Optional, Union

import torch.nn as nn
import transformers
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.models.registry import (_TRANSFORMERS_BACKEND_MODELS,
                                                 _LazyRegisteredModel,
                                                 _ModelInfo, _ModelRegistry,
                                                 _run)
from vllm.transformers_utils.dynamic_module import (
    try_get_class_from_dynamic_module)

from vllm_mindspore.utils import (is_mindformers_model_backend,
                                  is_mindone_model_backend,
                                  is_native_model_backend)

logger = init_logger(__name__)

try:
    from mindformers.tools.register.register import (MindFormerModuleType,
                                                     MindFormerRegister)
    mf_supported = True
    model_support_list = list(
        MindFormerRegister.registry[MindFormerModuleType.MODELS].keys())
    mcore_support_list = [
        name[len("mcore_"):] for name in model_support_list
        if name.startswith("mcore_")
    ]
except ImportError as e:
    logger.info("Can't get model support list from MindSpore Transformers: %s",
                e)
    if is_mindformers_model_backend():
        raise ImportError from e
    mf_supported = False
    mcore_support_list = []

try:
    from mindone import transformers  # noqa: F401
    mindone_supported = True
except ImportError as e:
    logger.info("No MindSpore ONE: %s", e)
    if is_mindone_model_backend():
        raise ImportError from e
    mindone_supported = False

_NATIVE_MODELS = {
    "MiniCPMForCausalLM": ("minicpm", "MiniCPMForCausalLM"),
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM"),
    "Qwen2_5_VLForConditionalGeneration":
    ("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration"),
    "Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),
    "Qwen3VLForConditionalGeneration":
    ("qwen3_vl", "Qwen3VLForConditionalGeneration"),
    "Qwen3VLMoeForConditionalGeneration":
    ("qwen3_vl_moe", "Qwen3VLMoeForConditionalGeneration"),
    "Glm4vForConditionalGeneration":
    ("glm4_1v", "Glm4vForConditionalGeneration"),
    "Glm4ForCausalLM": ("glm4", "Glm4ForCausalLM"),
}

_MINDFORMERS_MODELS = {
    "MindFormersForCausalLM": ("mindformers", "MindFormersForCausalLM")
}

_MINDONE_MODELS = {
    "TransformersForCausalLM": ("transformers", "TransformersForCausalLM"),
}
"""
Models with a fixed import path can be specified here to bypass the automatic 
backend selection. This is useful if you want to force a specific model to
always use a certain backend implementation.

Example:

AUTO_SELECT_FIXED_MODEL = {
    "Qwen3ForCausalLM": (
        "vllm_mindspore.model_executor.models.qwen3",  # module path
        "Qwen3ForCausalLM"                             # class name
    ),
}
"""
AUTO_SELECT_FIXED_MODEL = {}


def _register_model(backends: list[str], paths: list[str]):
    _registry_dict = {}
    for backend, model_dir in zip(backends, paths):
        if backend == _MINDFORMERS_MODELS:
            if not mf_supported:
                continue
        elif backend == _MINDONE_MODELS:  # noqa: SIM102
            if not mindone_supported:
                continue
        for model_arch, (mod_relname, cls_name) in backend.items():
            if model_arch not in _registry_dict:
                _registry_dict.update({
                    model_arch:
                    _LazyRegisteredModel(
                        module_name=model_dir.format(mod_relname),
                        class_name=cls_name,
                    )
                })
    return _registry_dict


if is_mindformers_model_backend():
    model_paths = "vllm_mindspore.model_executor.models.mf_models.{}"
    _registry_dict = _register_model([_MINDFORMERS_MODELS], [model_paths])
elif is_mindone_model_backend():
    model_paths = "vllm_mindspore.model_executor.models.mindone_models.{}"
    _registry_dict = _register_model([_MINDONE_MODELS], [model_paths])
elif is_native_model_backend():
    model_paths = "vllm_mindspore.model_executor.models.{}"
    _registry_dict = _register_model([_NATIVE_MODELS], [model_paths])
else:
    # mix backend selection, priority: mindformers > native > mindone
    model_backends = [_MINDFORMERS_MODELS, _NATIVE_MODELS, _MINDONE_MODELS]
    model_paths = [
        "vllm_mindspore.model_executor.models.mf_models.{}",
        "vllm_mindspore.model_executor.models.{}",
        "vllm_mindspore.model_executor.models.mindone_models.{}"
    ]
    _registry_dict = _register_model(model_backends, model_paths)

    # To override the auto selection result
    for arch in AUTO_SELECT_FIXED_MODEL:
        model_paths, cls_name = AUTO_SELECT_FIXED_MODEL[arch]
        _registry_dict.update({
            arch:
            _LazyRegisteredModel(
                module_name=model_paths,
                class_name=cls_name,
            )
        })

MindSporeModelRegistry = _ModelRegistry(_registry_dict)

_SUBPROCESS_COMMAND = [
    sys.executable, "-m", "vllm_mindspore.model_executor.models.registry"
]


def _try_resolve_transformers(
    self,
    architecture: str,
    model_config: ModelConfig,
) -> Optional[str]:
    if architecture in _TRANSFORMERS_BACKEND_MODELS:
        return architecture

    auto_map: dict[str, str] = getattr(model_config.hf_config, "auto_map",
                                       None) or dict()

    # Make sure that config class is always initialized before model class,
    # otherwise the model class won't be able to access the config class,
    # the expected auto_map should have correct order like:
    # Eg. "auto_map": {
    #     Eg. "AutoConfig": "<your-repo-name>--<config-name>",
    #     Eg. "AutoModel": "<your-repo-name>--<config-name>",
    #     Eg. "AutoModelFor<Task>": "<your-repo-name>--<config-name>",
    # },
    for prefix in ("AutoConfig", "AutoModel"):
        for name, module in auto_map.items():
            if name.startswith(prefix):
                try_get_class_from_dynamic_module(
                    module,
                    model_config.model,
                    revision=model_config.revision,
                    warn_on_fail=False,
                )

    model_module = getattr(transformers, architecture, None)

    if model_module is None:
        for name, module in auto_map.items():
            if name.startswith("AutoModel"):
                model_module = try_get_class_from_dynamic_module(
                    module,
                    model_config.model,
                    revision=model_config.revision,
                    warn_on_fail=True,
                )
                if model_module is not None:
                    break
        else:
            if model_config.model_impl != "transformers":
                return None

            raise ValueError(
                f"Cannot find model module. {architecture!r} is not a "
                "registered model in the Transformers library (only "
                "relevant if the model is meant to be in Transformers) "
                "and 'AutoModel' is not present in the model config's "
                "'auto_map' (relevant if the model is custom).")

    # vllm-ms: Bypass backend compatibility checks to
    # circumvent PyTorch validation.

    return model_config._get_transformers_backend_cls()


def _normalize_arch(
    self,
    architectures: Union[str, list[str]],
) -> list[str]:
    # Refer to
    # https://github.com/vllm-project/vllm/blob/releases/v0.9.2/vllm/model_executor/models/registry.py
    if isinstance(architectures, str):
        architectures = [architectures]
    if not architectures:
        logger.warning("No model architectures are specified")

    # filter out support architectures
    normalized_arch = list(
        filter(lambda model: model in self.models, architectures))

    # make sure MindFormersForCausalLM and MindONE Transformers backend
    # is put at the last as a fallback
    if len(normalized_arch) != len(architectures):
        normalized_arch.append("MindFormersForCausalLM")
        normalized_arch.append("TransformersForCausalLM")

    return normalized_arch


def inspect_model_cls(
    self,
    architectures: Union[str, list[str]],
    model_config: ModelConfig,
) -> tuple[_ModelInfo, str]:
    if isinstance(architectures, str):
        architectures = [architectures]
    if not architectures:
        raise ValueError("No model architectures are specified")

    # Require transformers impl
    if model_config.model_impl == "transformers":
        arch = self._try_resolve_transformers(architectures[0], model_config)
        if arch is not None:
            model_info = self._try_inspect_model_cls(arch)
            if model_info is not None:
                return (model_info, arch)
    elif model_config.model_impl == "terratorch":
        model_info = self._try_inspect_model_cls("Terratorch")
        return (model_info, "Terratorch")

    # Fallback to transformers impl (after resolving convert_type)
    if (all(arch not in self.models for arch in architectures)
            and model_config.model_impl == "auto"
            and getattr(model_config, "convert_type", "none") == "none"):
        arch = self._try_resolve_transformers(architectures[0], model_config)
        if arch is not None:
            model_info = self._try_inspect_model_cls(arch)
            if model_info is not None:
                return (model_info, arch)

    # vllm-ms: Modify the architecture normalization process in vLLM
    # to ensure that the MindFormers backend is added to the architecture.
    normalized_arch = self._normalize_arch(architectures)
    for arch in normalized_arch:
        model_info = self._try_inspect_model_cls(arch)
        if model_info is not None:
            return (model_info, arch)

    # Fallback to transformers impl (before resolving runner_type)
    if (all(arch not in self.models for arch in architectures)
            and model_config.model_impl == "auto"):
        arch = self._try_resolve_transformers(architectures[0], model_config)
        if arch is not None:
            model_info = self._try_inspect_model_cls(arch)
            if model_info is not None:
                return (model_info, arch)

    return self._raise_for_unsupported(architectures)


def resolve_model_cls(
    self,
    architectures: Union[str, list[str]],
    model_config: ModelConfig,
) -> tuple[type[nn.Module], str]:
    if isinstance(architectures, str):
        architectures = [architectures]
    if not architectures:
        raise ValueError("No model architectures are specified")

    # Require transformers impl
    if model_config.model_impl == "transformers":
        arch = self._try_resolve_transformers(architectures[0], model_config)
        if arch is not None:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)
    elif model_config.model_impl == "terratorch":
        arch = "Terratorch"
        model_cls = self._try_load_model_cls(arch)
        if model_cls is not None:
            return (model_cls, arch)

    # Fallback to transformers impl (after resolving convert_type)
    if (all(arch not in self.models for arch in architectures)
            and model_config.model_impl == "auto"
            and getattr(model_config, "convert_type", "none") == "none"):
        arch = self._try_resolve_transformers(architectures[0], model_config)
        if arch is not None:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)

    # vllm-ms: Modify the architecture normalization process in vLLM
    # to ensure that the MindFormers backend is added to the architecture.
    normalized_arch = self._normalize_arch(architectures)
    for arch in normalized_arch:
        model_cls = self._try_load_model_cls(arch)
        if model_cls is not None:
            return (model_cls, arch)

    # Fallback to transformers impl (before resolving runner_type)
    if (all(arch not in self.models for arch in architectures)
            and model_config.model_impl == "auto"):
        arch = self._try_resolve_transformers(architectures[0], model_config)
        if arch is not None:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)

    return self._raise_for_unsupported(architectures)


if __name__ == "__main__":
    _run()
