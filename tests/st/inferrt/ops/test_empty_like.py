# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Tests for torch.empty_like operation."""
import pytest
import torch

from ms_inferrt.torch import backend as fx_backend

from tests.mark_utils import arg_mark


def get_empty_like_compiled():
    def custom_op_func(x):
        return torch.empty_like(x)

    return torch.compile(custom_op_func, backend=fx_backend)


def get_empty_like_with_dtype_compiled():
    def custom_op_func(x, dtype):
        return torch.empty_like(x, dtype=dtype)

    return torch.compile(custom_op_func, backend=fx_backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [[10, 10], [2, 3, 4], []])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_empty_like_inherit_meta(shape, dtype):
    """
    Feature: Test torch.empty_like
    Description: Test empty_like inherits shape, dtype, and device from input tensor.
    Expectation: The output metadata is correct.
    """
    x = torch.randn(shape, dtype=dtype).npu()
    npu_output = get_empty_like_compiled()(x)

    assert npu_output.shape == x.shape
    assert npu_output.dtype == x.dtype
    assert npu_output.device.type == "npu"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [[8, 16], [1, 2, 3]])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32, torch.int8, torch.int16, torch.int32, torch.int64])
def test_empty_like_dtype_override(shape, out_dtype):
    """
    Feature: Test torch.empty_like
    Description: Test empty_like with explicit dtype override.
    Expectation: The output shape and device follow input, and dtype follows the argument.
    """
    x = torch.randn(shape, dtype=torch.float16).npu()
    npu_output = get_empty_like_with_dtype_compiled()(x, out_dtype)

    assert npu_output.shape == x.shape
    assert npu_output.dtype == out_dtype
    assert npu_output.device.type == "npu"
