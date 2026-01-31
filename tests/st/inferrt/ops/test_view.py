# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Tests for torch.view operation."""
import numpy as np
import pytest
import torch

from torch_npu.testing.common_utils import create_common_tensor

from mrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(input1, input2, shape):
    """op function for view with multiple inputs (no scalar temp)."""
    # Real computation before view: combine two tensor inputs
    x = input1 - input2
    # view operation
    y = x.view(shape)
    # Real computation after view: again combine with second input (reshaped)
    input2_view = input2.view(shape)
    z = y + input2_view
    return z


def view_forward(shape_format, op_func_compiled):
    """
    view forward function
    Args:
        shape_format: list of [dtype, format, shape]
        op_func_compiled: The compiled op function.
    """
    for item in shape_format:
        # create two inputs with same dtype/format/shape
        cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
        cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
        shape = [4, 16]
        cpu_output_view = op_func(cpu_input1, cpu_input2, shape)
        cpu_output = cpu_output_view.detach().numpy()
        npu_output_view = op_func_compiled(npu_input1, npu_input2, shape)
        npu_output = npu_output_view.detach().cpu().numpy()
        assert cpu_output_view.shape == npu_output_view.shape, "shape does match"
        assert cpu_output_view.stride() == npu_output_view.stride(), "stride does not match"
        AssertRtolEqual(cpu_output, npu_output)


# pylint: disable=redefined-builtin
def op_func_twice(input, shape1, shape2):
    """op function for view with two consecutive operations"""
    # First view operation
    temp = input.view(shape1)
    # Second view operation
    return temp.view(shape2)


def view_forward_twice(shape_format, op_func_compiled):
    """
    view forward function
    Args:
        shape_format: list of [dtype, format, shape]
        op_func_compiled: The compiled op function.
    """
    for item in shape_format:
        cpu_input, npu_input = create_common_tensor(item, 0, 100)
        shape1 = [4, 16]  # First view shape
        shape2 = [8, 8]   # Second view shape
        cpu_output = op_func_twice(cpu_input, shape1, shape2).detach().numpy()
        npu_output = op_func_compiled(npu_input, shape1, shape2).detach().cpu().numpy()
        AssertRtolEqual(cpu_output, npu_output)


def op_func_variadic(input_tensor, *shape):
    """op function for view with variadic int arguments."""
    return input_tensor.view(*shape)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_view():
    """
    Feature: Test view
    Description: Test view op with mlir_backend
    Expectation: The result is correct
    """
    dtype_list = [np.float16, np.float32, np.int32]
    format_list = [0]
    shape_list = [[8, 8], [2, 4, 8], [2, 4, 4, 2]]

    shape_format = [
        [i, j, k] for i in dtype_list for j in format_list for k in shape_list
    ]
    op_func_compiled = torch.compile(op_func, backend=backend)
    view_forward(shape_format, op_func_compiled)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_view_twice():
    """
    Feature: Test view
    Description: Test view op with mlir_backend
    Expectation: The result is correct
    """
    dtype_list = [np.float16, np.float32, np.int32, np.bool_]
    format_list = [0]
    shape_list = [[8, 8], [2, 4, 8], [2, 4, 4, 2]]

    shape_format = [
        [i, j, k] for i in dtype_list for j in format_list for k in shape_list
    ]
    op_func_compiled = torch.compile(op_func_twice, backend=backend)
    view_forward_twice(shape_format, op_func_compiled)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(64,), (4, 4, 4), (8, 8), (2, 16, 2)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_view_variadic_args(shape, dtype):
    """
    Feature: Test view with variadic int arguments
    Description: Test view op with single or multiple int arguments using torch.randn
    Expectation: The result is correct
    """

    cpu_input = torch.randn(8, 8, dtype=dtype)
    npu_input = cpu_input.clone().npu()

    op_func_compiled = torch.compile(op_func_variadic, backend=backend)
    cpu_output = op_func_variadic(cpu_input, *shape).detach().numpy()
    npu_output = op_func_compiled(npu_input, *shape).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output)
