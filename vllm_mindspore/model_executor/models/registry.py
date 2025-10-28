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
from typing import Union

from vllm.logger import init_logger
from vllm.model_executor.models.registry import (_LazyRegisteredModel,
                                                 _ModelRegistry, _run)

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
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM"),
    "Qwen2_5_VLForConditionalGeneration":
    ("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration"),
    "Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),
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


def _normalize_archs(
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


if __name__ == "__main__":
    _run()
