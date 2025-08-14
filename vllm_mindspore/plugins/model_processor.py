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

def apply_model_patch():

    import vllm.model_executor
    from vllm_mindspore.model_executor.model_loader.utils import (
        get_ms_model_architecture, )

    from vllm.model_executor.model_loader import get_model_architecture  # noqa F401
    vllm.model_executor.model_loader.get_model_architecture = (
        get_ms_model_architecture)
    vllm.model_executor.model_loader.utils.get_model_architecture = (
        get_ms_model_architecture)

    from vllm_mindspore.model_executor.models.registry import _normalize_archs
    from vllm.model_executor.models.registry import _ModelRegistry
    _ModelRegistry._normalize_archs = _normalize_archs
