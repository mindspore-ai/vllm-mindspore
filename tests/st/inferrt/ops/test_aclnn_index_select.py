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
"""Tests for aclnn index_select operation."""
import numpy as np
import pytest
import torch

from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(input_tensor, dim, index):
    """op function for index_select"""
    return torch.index_select(input_tensor, dim=dim, index=index)


def get_op_func_compiled():
    def custom_op_func(input_tensor, dim, index):
        return torch.index_select(input_tensor, dim=dim, index=index)
    return torch.compile(custom_op_func, backend=backend)


def index_select_forward(dtype, shape, dim, index, compiled_func):
    """
    index_select forward function
    Args:
        dtype: The data type of the input.
        shape: The shape of input tensor.
        dim: The dimension to select from.
        index: The index tensor.
        compiled_func: The compiled op function.
    """
    if np.issubdtype(dtype, np.integer):
        cpu_input = np.random.randint(-100, 100, shape).astype(dtype)
        prec = 0
    else:
        if dtype == np.float16:
            prec = 0.001
        else:
            prec = 0.0001
        cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)

    npu_input = torch.from_numpy(cpu_input).npu()
    npu_index = index.npu()

    cpu_output = op_func(torch.from_numpy(cpu_input), dim, index).numpy()
    npu_output = compiled_func(npu_input, dim, npu_index).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output, prec)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", (np.float32, np.float16))
@pytest.mark.parametrize("shape, dim, index", [
    ((3,), 0, torch.tensor([0, 1], dtype=torch.int64)),
    ((2, 4), 1, torch.tensor([0, 1, 2], dtype=torch.int64)),
    ((3, 4, 6), 2, torch.tensor([1, 2, 4], dtype=torch.int64)),
    ((4, 5, 6, 7), 3, torch.tensor([3, 5, 6], dtype=torch.int64)),
    ((3, 4, 8, 9, 12), 4, torch.tensor([2, 3, 5, 6], dtype=torch.int64)),
    ((3, 4, 8, 9, 12), -1, torch.tensor([2, 3, 5, 6], dtype=torch.int64)),
])
def test_index_select_float(dtype, shape, dim, index):
    """
    Feature: Test aclnn index_select
    Description: Test aclnn index_select with fp32/fp16 inputs and various shapes/dimensions
    Expectation: The result is correct
    """
    compiled_op = get_op_func_compiled()
    index_select_forward(dtype, shape, dim, index, compiled_op)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", (np.int8, np.int16, np.int32, np.int64))
@pytest.mark.parametrize("shape, dim, index", [
    ((3,), 0, torch.tensor([0, 1], dtype=torch.int64)),
    ((2, 4), 1, torch.tensor([0, 1, 2], dtype=torch.int64)),
    ((3, 4, 6), 2, torch.tensor([1, 2, 4], dtype=torch.int64)),
    ((4, 5, 6, 7), 3, torch.tensor([3, 5, 6], dtype=torch.int64)),
    ((3, 4, 8, 9, 12), 4, torch.tensor([2, 3, 5, 6], dtype=torch.int64)),
    ((3, 4, 8, 9, 12), -1, torch.tensor([2, 3, 5, 6], dtype=torch.int64)),
])
def test_index_select_int(dtype, shape, dim, index):
    """
    Feature: Test aclnn index_select
    Description: Test aclnn index_select with integer inputs and various shapes/dimensions
    Expectation: The result is correct
    """
    compiled_op = get_op_func_compiled()
    index_select_forward(dtype, shape, dim, index, compiled_op)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", (np.float32, np.float16))
@pytest.mark.parametrize("shape, dim, index", [
    ((3,), 0, torch.tensor(0, dtype=torch.int64)),
    ((2, 4), 1, torch.tensor(1, dtype=torch.int64)),
    ((3, 4, 6), 2, torch.tensor(2, dtype=torch.int64)),
])
def test_index_select_single_index(dtype, shape, dim, index):
    """
    Feature: Test aclnn index_select
    Description: Test aclnn index_select with single index
    Expectation: The result is correct
    """
    compiled_op = get_op_func_compiled()
    index_select_forward(dtype, shape, dim, index, compiled_op)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", (np.float32, np.float16))
def test_index_select_tensor_method(dtype):
    """
    Feature: Test aclnn index_select
    Description: Test aclnn index_select using tensor method
    Expectation: The result is correct
    """
    cpu_input = torch.from_numpy(np.random.rand(15, 25).astype(dtype))
    cpu_index = torch.from_numpy(np.array([1, 5, 10, 14]))

    npu_input = cpu_input.npu()
    npu_index = cpu_index.npu()

    cpu_output = cpu_input.index_select(0, cpu_index).numpy()

    def tensor_method_func(input_tensor, index):
        return input_tensor.index_select(0, index)

    compiled_func = torch.compile(tensor_method_func, backend=backend)
    npu_output = compiled_func(npu_input, npu_index).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output)
