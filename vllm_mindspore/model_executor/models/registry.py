#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import pickle
import subprocess
import sys
import tempfile
from typing import Callable, TypeVar

import cloudpickle

from vllm.model_executor.models.registry import _ModelRegistry, _LazyRegisteredModel

from vllm_mindspore.utils import is_mindformers_model_backend

_MINDSPORE_MODELS = {
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
}

_MINDFORMERS_MODELS = {
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "DeepseekV3ForCausalLM": ("deepseek_v3", "DeepseekV3ForCausalLM"),
    "DeepSeekMTPModel": ("deepseek_mtp", "DeepseekV3MTPForCausalLM"),
}

MindSporeModelRegistry = _ModelRegistry(
    {
        model_arch: _LazyRegisteredModel(
            module_name=f"vllm_mindspore.model_executor.models.{mod_relname}",
            class_name=cls_name,
        )
        for model_arch, (mod_relname, cls_name) in _MINDSPORE_MODELS.items()
    }
    if not is_mindformers_model_backend()
    else {
        model_arch: _LazyRegisteredModel(
            module_name=f"vllm_mindspore.model_executor.models.mf_models.{mod_relname}",
            class_name=cls_name,
        )
        for model_arch, (mod_relname, cls_name) in _MINDFORMERS_MODELS.items()
    }
)

_T = TypeVar("_T")


_SUBPROCESS_COMMAND = [
    sys.executable, "-m", "vllm_mindspore.model_executor.models.registry"
]


def _run() -> None:
    # Setup plugins
    from vllm.plugins import load_general_plugins
    load_general_plugins()

    fn, output_file = pickle.loads(sys.stdin.buffer.read())

    result = fn()

    with open(output_file, "wb") as f:
        f.write(pickle.dumps(result))


if __name__ == "__main__":
    _run()
