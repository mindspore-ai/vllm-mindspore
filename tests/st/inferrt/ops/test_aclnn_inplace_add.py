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

"""Test module for aclnn inplace_add operation."""

import pytest
import numpy as np
import torch

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual
from mrt.torch import backend


def op_func_alpha_2(x1, x2):
    """op function for inplace_add with alpha=2"""
    x1.add_(x2, alpha=2)
    return x1


def op_func_alpha_0_5(x1, x2):
    """op function for inplace_add with alpha=0.5"""
    x1.add_(x2, alpha=0.5)
    return x1


def op_func_no_alpha(x1, x2):
    """op function without alpha, using += operator"""
    x1 += x2
    return x1


def op_func_no_alpha_scalar_3_6(x1, _):
    """op function without alpha, using += operator"""
    x1 += 3.6
    return x1


def op_func_no_alpha_scalar_12(x1, _):
    """op function without alpha, using += operator"""
    x1 += 12
    return x1


def inplace_add_forward(dtype, shape, op_func, op_func_compiled, _, other_type, other_value=None):
    """
    inplace_add forward function
    Args:
        dtype: The data type of the input.
        shape: The shape of the input tensor.
        op_func: The original op function for CPU.
        op_func_compiled: The compiled op function for NPU.
        _alpha: Alpha parameter (unused, kept for compatibility).
        other_type: Type of the second operand ('tensor', 'scalar', 'cpu_scalar_tensor', 'npu_scalar_tensor').
        other_value: Value or tensor to add.
    """
    cpu_output = None
    npu_output = None

    if np.issubdtype(dtype, np.integer):
        cpu_input = np.random.randint(-100, 100, shape).astype(dtype)
        prec = 0
    else:
        if dtype == np.float16:
            prec = 0.001
        else:
            prec = 0.0001
        cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)

    cpu_input = torch.from_numpy(cpu_input)
    npu_input = cpu_input.npu()

    if other_type == 'tensor':
        if np.issubdtype(dtype, np.integer):
            cpu_other = np.random.randint(-100, 100, shape).astype(dtype)
        else:
            cpu_other = np.random.uniform(-1, 1, shape).astype(dtype)

        cpu_other = torch.from_numpy(cpu_other)
        npu_other = cpu_other.npu()
        cpu_output = op_func(cpu_input, cpu_other)
        npu_output = op_func_compiled(npu_input, npu_other)
    elif other_type == 'scalar':
        cpu_output = op_func(cpu_input, other_value)
        npu_output = op_func_compiled(npu_input, other_value)
    elif other_type == 'cpu_scalar_tensor':
        cpu_other_tensor = torch.from_numpy(np.array(other_value, dtype=dtype))
        cpu_output = op_func(cpu_input, cpu_other_tensor)
        npu_output = op_func_compiled(npu_input, cpu_other_tensor)
    elif other_type == 'npu_scalar_tensor':
        cpu_other_tensor = torch.from_numpy(np.array(other_value, dtype=dtype))
        npu_other_tensor = cpu_other_tensor.npu()
        cpu_output = op_func(cpu_input, cpu_other_tensor)
        npu_output = op_func_compiled(npu_input, npu_other_tensor)

    AssertRtolEqual(cpu_output.detach().numpy(), npu_output.cpu().numpy(), prec)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(1024, 1024), (256, 512)])
@pytest.mark.parametrize("dtype,op_func,alpha", [
    (np.float32, op_func_alpha_2, 2),
    (np.float32, op_func_alpha_0_5, 0.5),
    (np.float32, op_func_no_alpha, 1.0),
    (np.int32, op_func_alpha_2, 2),
    (np.int32, op_func_no_alpha, 1)
])
@pytest.mark.parametrize("other_type", ['tensor'])
@pytest.mark.parametrize("other_value", [None])
def test_inplace_add_tensor_tensor(dtype, shape, op_func, alpha, other_type, other_value):
    """
    Feature: Test aclnn inplace_add
    Description: Test aclnn inplace_add with tensor + tensor
    Expectation: The result is correct
    """
    op_func_compiled = torch.compile(op_func, backend=backend)
    inplace_add_forward(dtype, shape, op_func, op_func_compiled, alpha, other_type, other_value)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(1024, 1024), (256, 512)])
@pytest.mark.parametrize("dtype,op_func,alpha,other_value", [
    (np.float32, op_func_no_alpha_scalar_3_6, 1.0, 3.6),
    (np.int32, op_func_no_alpha_scalar_12, 1, 12)
])
@pytest.mark.parametrize("other_type", ['scalar'])
def test_inplace_add_tensor_scalar(dtype, shape, op_func, alpha, other_type, other_value):
    """
    Feature: Test aclnn inplace_add
    Description: Test aclnn inplace_add with tensor + scalar
    Expectation: The result is correct
    """
    op_func_compiled = torch.compile(op_func, backend=backend)
    inplace_add_forward(dtype, shape, op_func, op_func_compiled, alpha, other_type, other_value)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(1024, 1024), (256, 512)])
@pytest.mark.parametrize("dtype,op_func,alpha,other_value", [
    (np.float32, op_func_alpha_2, 2, 3.6),
    (np.float32, op_func_alpha_0_5, 0.5, 3.6),
    (np.float32, op_func_no_alpha, 1.0, 3.6),
    (np.int32, op_func_alpha_2, 2, 12),
    (np.int32, op_func_no_alpha, 1, 12)
])
@pytest.mark.parametrize("other_type", ['npu_scalar_tensor'])
def test_inplace_add_tensor_npu_scalar_tensor(dtype, shape, op_func, alpha, other_type, other_value):
    """
    Feature: Test aclnn inplace_add
    Description: Test aclnn inplace_add with tensor + npu scalar tensor
    Expectation: The result is correct
    """
    op_func_compiled = torch.compile(op_func, backend=backend)
    inplace_add_forward(dtype, shape, op_func, op_func_compiled, alpha, other_type, other_value)
