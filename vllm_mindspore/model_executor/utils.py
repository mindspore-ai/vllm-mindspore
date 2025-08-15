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
import mindspore as ms
import torch

TORCH_DTYPE_TO_MS_DTYPE = {
    torch.bfloat16: ms.bfloat16,
    torch.float16: ms.float16,
    torch.float32: ms.float32,
    torch.float64: ms.float64,
    torch.int8: ms.int8,
    torch.int16: ms.int16,
    torch.int32: ms.int32,
    torch.int64: ms.int64,
    torch.int64: ms.int64,
    torch.uint8: ms.uint8,
    torch.bool: ms.bool_,
    torch.long: ms.int64,
    torch.half: ms.float16,
    torch.int: ms.int32,
    torch.double: ms.float64,
    torch.float: ms.float32,
    torch.short: ms.int16
}


def get_ms_dtype(torch_dtype):
    return TORCH_DTYPE_TO_MS_DTYPE.get(torch_dtype)

def tensor_torch2ms(x: torch.Tensor):
    if x is None or not isinstance(x, torch.Tensor):
        return x

    if x.device.type == "cpu":
        # TODO: dlpack support CPU, for now will slow down the weight loading
        if x.dtype == torch.bfloat16:
            return ms.Tensor(
                x.contiguous().to(torch.float32).numpy(), dtype=ms.bfloat16
            )
        return ms.Tensor(x.contiguous().numpy())

    # torch tensor -> dlpack -> mindspore tensor
    pt_dlpack = torch.utils.dlpack.to_dlpack(x)
    ms_tensor = ms.Tensor.from_dlpack(pt_dlpack)
    return ms_tensor


def tensor_ms2torch(x: ms.Tensor):
    if x is None or not isinstance(x, ms.Tensor):
        return x

    if x.device == "CPU":  # TODO: dlpack support CPU
        if x.dtype == ms.bfloat16:
            return torch.tensor(
                x.contiguous().to(ms.float32).asnumpy(), dtype=torch.bfloat16
            )
        return torch.tensor(x.contiguous().asnumpy())

    # ms tensor -> dlpack -> torch tensor
    ms_dlpack = x.to_dlpack()
    torch_tensor = torch.from_dlpack(ms_dlpack)
    return torch_tensor


def set_weight_attrs(
    weight: Tensor,
    weight_attrs: Optional[dict[str, Any]],
):
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        setattr(weight, key, value)


_native_model_context = {"is_prefill": True}


def set_model_context(key, value):
    global _native_model_context
    _native_model_context[key] = value


def get_model_context(key):
    return _native_model_context[key]
