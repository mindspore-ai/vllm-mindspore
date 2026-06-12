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
"""Tests for eq operation."""
import numpy as np
import pytest
import torch

from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(input1, input2):
    """golden for eq"""
    return torch.eq(input1, input2)


def get_op_func_compiled():
    """eq op compiled"""
    def custom_op_func(x, y):
        return torch.eq(x, y)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8, 16), (64, 32), (128, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_eq(shape, dtype):
    """
    Feature: Test op eq
    Description: Test op eq with multi-dimensional tensor inputs
    Expectation: The result is correct
    """
    cpu_input1 = torch.rand(shape, dtype=dtype)
    cpu_input2 = torch.rand(shape, dtype=dtype)
    npu_input1 = cpu_input1.npu()
    npu_input2 = cpu_input2.npu()
    cpu_output = op_func(cpu_input1, cpu_input2)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(npu_input1, npu_input2)
    npu_output_cpu = npu_output.cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)

def aten_eq_scalar_rhs_op(x, scalar_val):
    return torch.ops.aten.eq.Scalar(x, scalar_val)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8,), (4, 8), (16, 32), (64, 128), (2, 4, 8)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int32])
def test_aten_eq_scalar_rhs(shape, dtype):
    """
Feature: Test aten eq scalar with scalar on RHS.
    Description: Test aten.eq.Scalar(x, 0) with various shapes.
    Expectation: The result matches eager mode.
    """
    compiled_op = torch.compile(aten_eq_scalar_rhs_op, backend=backend)
    scalar_val = 0
    if np.issubdtype(dtype, np.integer):
        cpu_input = np.random.randint(-10, 10, shape).astype(dtype)
    else:
        cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    npu_input = torch.from_numpy(cpu_input).npu()
    cpu_output = aten_eq_scalar_rhs_op(torch.from_numpy(cpu_input), scalar_val).detach().numpy()
    npu_output = compiled_op(npu_input, scalar_val).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output)

def aten_eq_scalar_lhs_op(scalar_val, x):
    return torch.ops.aten.eq.Scalar(x, scalar_val)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8,), (4, 8), (16, 32), (64, 128), (2, 4, 8)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int32])
def test_aten_eq_scalar_lhs(shape, dtype):
    """
Feature: Test aten eq scalar with scalar on LHS.
    Description: Test eq with scalar on left side.
    Expectation: The result matches eager mode.
    """
    compiled_op = torch.compile(aten_eq_scalar_lhs_op, backend=backend)
    scalar_val = 0
    if np.issubdtype(dtype, np.integer):
        cpu_input = np.random.randint(-10, 10, shape).astype(dtype)
    else:
        cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    npu_input = torch.from_numpy(cpu_input).npu()
    cpu_output = torch.ops.aten.eq.Scalar(torch.from_numpy(cpu_input), scalar_val).detach().numpy()
    npu_output = compiled_op(scalar_val, npu_input).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output)

def eq_tensor_dynamic_op(x, y):
    b = x.size(0)
    return torch.eq(x[:b], y[:b])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8,), (4, 8), (16, 32), (64, 128), (2, 4, 8)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int32])
def test_eq_tensor_dynamic(shape, dtype):
    """
Feature: Test eq tensor with dynamic shapes.
    Description: Test torch.eq with two tensor inputs and dynamic slicing.
    Expectation: The result matches eager mode.
    """
    compiled_op = torch.compile(eq_tensor_dynamic_op, backend=backend)
    if np.issubdtype(dtype, np.integer):
        cpu_input0 = np.random.randint(-10, 10, shape).astype(dtype)
        cpu_input1 = np.random.randint(-10, 10, shape).astype(dtype)
    else:
        cpu_input0 = np.random.uniform(-1, 1, shape).astype(dtype)
        cpu_input1 = np.random.uniform(-1, 1, shape).astype(dtype)
    npu_input0 = torch.from_numpy(cpu_input0).npu()
    npu_input1 = torch.from_numpy(cpu_input1).npu()
    cpu_output = eq_tensor_dynamic_op(torch.from_numpy(cpu_input0), torch.from_numpy(cpu_input1)).detach().numpy()
    npu_output = compiled_op(npu_input0, npu_input1).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8,), (4, 8), (16, 32), (64, 128)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_eq_tensor_static(shape, dtype):
    """
Feature: Test eq tensor with static shapes.
    Description: Test torch.eq with two tensor inputs.
    Expectation: The result matches eager mode.
    """
    def eq_tensor_static_op(x, y):
        return torch.eq(x, y)

    compiled_op = torch.compile(eq_tensor_static_op, backend=backend)
    cpu_input0 = np.random.uniform(-1, 1, shape).astype(dtype)
    cpu_input1 = np.random.uniform(-1, 1, shape).astype(dtype)
    npu_input0 = torch.from_numpy(cpu_input0).npu()
    npu_input1 = torch.from_numpy(cpu_input1).npu()
    cpu_output = eq_tensor_static_op(torch.from_numpy(cpu_input0), torch.from_numpy(cpu_input1)).detach().numpy()
    npu_output = compiled_op(npu_input0, npu_input1).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output)



def eq_scalar_rhs_op(x, scalar_val):
    return torch.eq(x, scalar_val)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8,), (4, 8), (16, 32), (64, 128), (2, 4, 8)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int32])
def test_eq_scalar_rhs(shape, dtype):
    """
Feature: Test eq scalar on RHS with dynamic shapes.
    Description: Test torch.eq(x, 0) with various shapes.
    Expectation: The result matches eager mode.
    """
    compiled_op = torch.compile(eq_scalar_rhs_op, backend=backend)
    scalar_val = 0
    if np.issubdtype(dtype, np.integer):
        cpu_input = np.random.randint(-10, 10, shape).astype(dtype)
    else:
        cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    npu_input = torch.from_numpy(cpu_input).npu()
    cpu_output = eq_scalar_rhs_op(torch.from_numpy(cpu_input), scalar_val).detach().numpy()
    npu_output = compiled_op(npu_input, scalar_val).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output)


def eq_scalar_lhs_op(scalar_val, x):
    return scalar_val == x


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8,), (4, 8), (16, 32), (64, 128), (2, 4, 8)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int32])
def test_eq_scalar_lhs(shape, dtype):
    """
Feature: Test eq scalar on LHS with dynamic shapes.
    Description: Test (0 == x) with various shapes.
    Expectation: The result matches eager mode.
    """
    compiled_op = torch.compile(eq_scalar_lhs_op, backend=backend)
    scalar_val = 0
    if np.issubdtype(dtype, np.integer):
        cpu_input = np.random.randint(-10, 10, shape).astype(dtype)
    else:
        cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    npu_input = torch.from_numpy(cpu_input).npu()
    # eq is symmetric: scalar == tensor equals tensor == scalar
    cpu_output = torch.eq(torch.from_numpy(cpu_input), scalar_val).detach().numpy()
    npu_output = compiled_op(scalar_val, npu_input).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output)
