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


def _register_ms_models():
    import vllm.config
    from vllm_mindspore.model_executor.models.registry import (
        MindSporeModelRegistry,
        _SUBPROCESS_COMMAND,
    )
    from vllm_mindspore.model_executor.layers.quantization import (
        get_quantization_config)

    vllm.config.ModelRegistry = MindSporeModelRegistry

    import vllm.model_executor

    vllm.model_executor.models.ModelRegistry = MindSporeModelRegistry

    from vllm_mindspore.model_executor.model_loader.utils import (
        get_ms_model_architecture, ms_device_loading_context)

    # To patching the get_model_architecture, should import it first.
    from vllm.model_executor.model_loader import get_model_architecture  # noqa F401

    vllm.model_executor.model_loader.get_model_architecture = (
        get_ms_model_architecture)
    vllm.model_executor.model_loader.utils.get_model_architecture = (
        get_ms_model_architecture)
    vllm.model_executor.model_loader.default_loader.get_model_architecture = (
        get_ms_model_architecture)
    vllm.model_executor.model_loader.utils.device_loading_context = (
        ms_device_loading_context)

    from vllm_mindspore.model_executor.models.registry import (
        _normalize_arch, _try_resolve_transformers, inspect_model_cls,
        resolve_model_cls)
    from vllm.model_executor.models.registry import _ModelRegistry

    _ModelRegistry._normalize_arch = _normalize_arch
    _ModelRegistry._try_resolve_transformers = _try_resolve_transformers
    _ModelRegistry.inspect_model_cls = inspect_model_cls
    _ModelRegistry.resolve_model_cls = resolve_model_cls
