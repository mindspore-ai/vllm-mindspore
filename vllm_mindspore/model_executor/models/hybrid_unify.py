# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/model_executor/models/utils.py
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
""" This file can only be referenced by native models."""

import os
import numpy as np

import mindspore as ms

import vllm_mindspore.envs as env

if not env.HYBRID_MODE:
    from vllm_mindspore.model_executor.models.model_base import NativeModel
    UnifiedNativeModel = NativeModel
else:
    from vllm_mindspore.hybrid_adapter.model_base import MsModelAdapter
    UnifiedNativeModel = MsModelAdapter


def get_ms_tensor(input_data, dtype=None):
    if input_data is None or isinstance(input_data, ms.Tensor):
        return input_data
    elif isinstance(input_data, np.ndarray):
        if dtype is not None:
            return ms.Tensor(input_data, dtype=dtype)
        return ms.from_numpy(input_data)
    else:
        from vllm_mindspore.hybrid_adapter.tensor_convert import (
            tensor_torch2ms)
        return tensor_torch2ms(input_data)
