# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/model_executor/utils.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2025 The vLLM team.
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

from typing import Any, Optional

from mindspore import Tensor


def set_weight_attrs(
    weight: Tensor,
    weight_attrs: Optional[dict[str, Any]],
):
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        from vllm.platforms import current_platform
        """
        Ensure the parameter is materialized before loading weights,
        as required by deferred initialization to reduce peak memory.
        """
        if key == "weight_loader" and hasattr(
                current_platform, "make_materializing_weight_loader"):
            value = current_platform.make_materializing_weight_loader(value)
        setattr(weight, key, value)


_native_model_context = {"is_prefill": True}


def set_model_context(key, value):
    global _native_model_context
    _native_model_context[key] = value


def get_model_context(key):
    return _native_model_context[key]
