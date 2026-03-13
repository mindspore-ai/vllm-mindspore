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
"""Tests for tensor with storage offset input."""
import pytest
import torch

from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def sigmoid_func(x):
    """sigmoid function"""
    return torch.sigmoid(x)


def add_func(x, y):
    """add function"""
    return x + y


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(4, 5), (8, 16)])
@pytest.mark.parametrize("split_idx", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_input_offset_from_split(shape, split_idx, dtype):
    """
    Feature: Test tensor with storage offset from split
    Description: Test operations on view tensors created by torch.split
    Expectation: The result matches eager mode
    """
    base = torch.randn(shape, dtype=dtype).npu()
    splits = torch.split(base, shape[0] // 2, dim=0)
    x = splits[split_idx]

    result_eager = sigmoid_func(x)
    compiled_func = torch.compile(sigmoid_func, backend=backend)
    result_compiled = compiled_func(x)

    AssertRtolEqual(result_eager, result_compiled.cpu())


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8, 8), (16, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_input_offset_from_slice(shape, dtype):
    """
    Feature: Test tensor with storage offset from slice
    Description: Test operations on view tensors created by slicing
    Expectation: The result matches eager mode
    """
    base = torch.randn(shape, dtype=dtype).npu()
    x = base[2:5, :]

    result_eager = sigmoid_func(x)
    compiled_func = torch.compile(sigmoid_func, backend=backend)
    result_compiled = compiled_func(x)

    AssertRtolEqual(result_eager, result_compiled.cpu())


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(6, 8), (12, 16)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_input_offset_binary_op(shape, dtype):
    """
    Feature: Test binary operation with storage offset tensors
    Description: Test add operation on two view tensors with different storage offsets
    Expectation: The result matches eager mode
    """
    base = torch.randn(shape, dtype=dtype).npu()
    splits = torch.split(base, shape[0] // 2, dim=0)
    x = splits[0]
    y = splits[1]

    result_eager = add_func(x, y)
    compiled_func = torch.compile(add_func, backend=backend)
    result_compiled = compiled_func(x, y)

    AssertRtolEqual(result_eager, result_compiled.cpu())
