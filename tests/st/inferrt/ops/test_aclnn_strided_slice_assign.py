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

"""Tests for aclnn strided_slice_assign operation (tensor slice assignment x[...]=value)."""
import numpy as np
import pytest
import torch

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


# ============================================================================
# Slice assign op functions
# ============================================================================

def setitem_slice_basic_op(x, value):
    """Basic slice assign: x[1:5] = value, then return x"""
    x = x.clone()
    x[1:5] = value
    return x


def setitem_slice_end_only_op(x, value):
    """Slice assign with end only: x[:3] = value"""
    x = x.clone()
    x[:3] = value
    return x


def setitem_slice_start_only_op(x, value):
    """Slice assign with start only: x[5:] = value"""
    x = x.clone()
    x[5:] = value
    return x


def setitem_slice_with_step_op(x, value):
    """Slice assign with step: x[0:8:2] = value"""
    x = x.clone()
    x[0:8:2] = value
    return x


def setitem_slice_multi_dim_op(x, value):
    """Multi-dim slice assign: x[1:3, 2:5] = value"""
    x = x.clone()
    x[1:3, 2:5] = value
    return x


def setitem_slice_full_dim_op(x, value):
    """Full dim slice assign: x[:, 0:4] = value"""
    x = x.clone()
    x[:, 0:4] = value
    return x


def setitem_slice_3d_op(x, value):
    """3D slice assign: x[0:2, 1:3, 2:5] = value"""
    x = x.clone()
    x[0:2, 1:3, 2:5] = value
    return x


def setitem_slice_negative_op(x, value):
    """Negative indices slice assign: x[-3:-1] = value"""
    x = x.clone()
    x[-3:-1] = value
    return x


# ============================================================================
# Dynamic shape slice assign op functions
# ============================================================================

def setitem_slice_dynamic_op(x, value):
    """Dynamic slice assign: x[:b] = value where b = x.size(0)"""
    b = x.size(0)
    x = x.clone()
    x[:b] = value[:b]
    return x


def setitem_slice_dynamic_partial_op(x, value):
    """Dynamic partial slice assign: x[:b, :] = value"""
    b = x.size(0)
    x = x.clone()
    x[:b, :] = value[:b, :]
    return x


# ============================================================================
# Helper
# ============================================================================

# ============================================================================
# Static shape tests
# ============================================================================

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8,), (16,), (32,)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_strided_slice_assign_basic(shape, dtype):
    """
    Feature: Test aclnn strided_slice_assign
    Description: Test basic slice assignment x[1:5] = value with 1D tensors
    Expectation: The result is correct
    """
    compiled_op = torch.compile(setitem_slice_basic_op, backend=backend)

    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    cpu_value = np.random.uniform(0.5, 2, (4,)).astype(dtype)

    npu_input = torch.from_numpy(cpu_input).npu()
    npu_value = torch.from_numpy(cpu_value).npu()

    cpu_output = setitem_slice_basic_op(torch.from_numpy(cpu_input), torch.from_numpy(cpu_value)).detach().numpy()
    npu_output = compiled_op(npu_input, npu_value).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8, 16), (16, 32)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_strided_slice_assign_2d_basic(shape, dtype):
    """
    Feature: Test aclnn strided_slice_assign on 2D tensors
    Description: Test x[1:5] = value on 2D tensors
    Expectation: The result is correct
    """
    compiled_op = torch.compile(setitem_slice_basic_op, backend=backend)

    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    cpu_value = np.random.uniform(0.5, 2, (4, shape[1])).astype(dtype)

    npu_input = torch.from_numpy(cpu_input).npu()
    npu_value = torch.from_numpy(cpu_value).npu()

    cpu_output = setitem_slice_basic_op(torch.from_numpy(cpu_input), torch.from_numpy(cpu_value)).detach().numpy()
    npu_output = compiled_op(npu_input, npu_value).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_strided_slice_assign_with_step(dtype):
    """
    Feature: Test aclnn strided_slice_assign with step
    Description: Test x[0:8:2] = value assignment with step
    Expectation: The result is correct
    """
    shape = (12, 16)
    compiled_op = torch.compile(setitem_slice_with_step_op, backend=backend)

    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    # Step=2 on dim 0 from 0 to 8 gives 4 elements
    cpu_value = np.random.uniform(0.5, 2, (4, shape[1])).astype(dtype)

    npu_input = torch.from_numpy(cpu_input).npu()
    npu_value = torch.from_numpy(cpu_value).npu()

    cpu_output = setitem_slice_with_step_op(torch.from_numpy(cpu_input), torch.from_numpy(cpu_value)).detach().numpy()
    npu_output = compiled_op(npu_input, npu_value).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_strided_slice_assign_multi_dim(dtype):
    """
    Feature: Test aclnn strided_slice_assign on multi-dimensional slice
    Description: Test x[1:3, 2:5] = value assignment
    Expectation: The result is correct
    """
    shape = (8, 16)
    compiled_op = torch.compile(setitem_slice_multi_dim_op, backend=backend)

    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    cpu_value = np.random.uniform(0.5, 2, (2, 3)).astype(dtype)

    npu_input = torch.from_numpy(cpu_input).npu()
    npu_value = torch.from_numpy(cpu_value).npu()

    cpu_output = setitem_slice_multi_dim_op(torch.from_numpy(cpu_input), torch.from_numpy(cpu_value)).detach().numpy()
    npu_output = compiled_op(npu_input, npu_value).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_strided_slice_assign_full_dim(dtype):
    """
    Feature: Test aclnn strided_slice_assign with full dimension
    Description: Test x[:, 0:4] = value assignment on full first dim
    Expectation: The result is correct
    """
    shape = (8, 16)
    compiled_op = torch.compile(setitem_slice_full_dim_op, backend=backend)

    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    cpu_value = np.random.uniform(0.5, 2, (shape[0], 4)).astype(dtype)

    npu_input = torch.from_numpy(cpu_input).npu()
    npu_value = torch.from_numpy(cpu_value).npu()

    cpu_output = setitem_slice_full_dim_op(torch.from_numpy(cpu_input), torch.from_numpy(cpu_value)).detach().numpy()
    npu_output = compiled_op(npu_input, npu_value).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_strided_slice_assign_negative_indices(dtype):
    """
    Feature: Test aclnn strided_slice_assign with negative indices
    Description: Test x[-3:-1] = value assignment with negative indices
    Expectation: The result is correct
    """
    shape = (8, 16)
    compiled_op = torch.compile(setitem_slice_negative_op, backend=backend)

    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    # -3:-1 on dim 0 gives 2 elements
    cpu_value = np.random.uniform(0.5, 2, (2, shape[1])).astype(dtype)

    npu_input = torch.from_numpy(cpu_input).npu()
    npu_value = torch.from_numpy(cpu_value).npu()

    cpu_output = setitem_slice_negative_op(torch.from_numpy(cpu_input), torch.from_numpy(cpu_value)).detach().numpy()
    npu_output = compiled_op(npu_input, npu_value).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_strided_slice_assign_3d(dtype):
    """
    Feature: Test aclnn strided_slice_assign on 3D tensors
    Description: Test x[0:2, 1:3, 2:5] = value assignment on 3D tensor
    Expectation: The result is correct
    """
    shape = (4, 8, 16)
    compiled_op = torch.compile(setitem_slice_3d_op, backend=backend)

    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    cpu_value = np.random.uniform(0.5, 2, (2, 2, 3)).astype(dtype)

    npu_input = torch.from_numpy(cpu_input).npu()
    npu_value = torch.from_numpy(cpu_value).npu()

    cpu_output = setitem_slice_3d_op(torch.from_numpy(cpu_input), torch.from_numpy(cpu_value)).detach().numpy()
    npu_output = compiled_op(npu_input, npu_value).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output)


# ============================================================================
# bfloat16 tests
# ============================================================================

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8, 16), (16, 32)])
def test_strided_slice_assign_bfloat16(shape):
    """
    Feature: Test aclnn strided_slice_assign with bfloat16
    Description: Test slice assignment with bf16 dtype
    Expectation: The result is correct
    """
    dtype = torch.bfloat16

    def setitem_bf16_op(x, value):
        x = x.clone()
        x[1:5] = value
        return x

    compiled_op = torch.compile(setitem_bf16_op, backend=backend)

    npu_input = torch.randn(shape, dtype=dtype).npu()
    npu_value = torch.randn((4, shape[1]), dtype=dtype).npu()

    # Eager reference on NPU
    input_ref = npu_input.clone()
    value_ref = npu_value.clone()
    expected = setitem_bf16_op(input_ref, value_ref)

    npu_output = compiled_op(npu_input, npu_value)
    AssertRtolEqual(expected.detach().cpu(), npu_output.detach().cpu(), 0.001)


# ============================================================================
# Dynamic shape tests
# ============================================================================

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8,), (4, 8), (16, 32), (64, 128), (2, 4, 8)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_strided_slice_assign_dynamic(shape, dtype):
    """
    Feature: Test aclnn strided_slice_assign with dynamic shape
    Description: Test x[:b] = value dynamic slice assignment pattern
    Expectation: The result is correct
    """
    compiled_op = torch.compile(setitem_slice_dynamic_op, backend=backend)

    if len(shape) == 1:
        cpu_value_shape = shape
    else:
        cpu_value_shape = shape

    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    cpu_value = np.random.uniform(0.5, 2, cpu_value_shape).astype(dtype)

    npu_input = torch.from_numpy(cpu_input).npu()
    npu_value = torch.from_numpy(cpu_value).npu()

    cpu_output = setitem_slice_dynamic_op(torch.from_numpy(cpu_input), torch.from_numpy(cpu_value)).detach().numpy()
    npu_output = compiled_op(npu_input, npu_value).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(4, 8), (16, 32), (64, 128)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_strided_slice_assign_dynamic_partial(shape, dtype):
    """
    Feature: Test aclnn strided_slice_assign with partial dynamic shape
    Description: Test x[:b, :] = value dynamic slice assignment on 2D
    Expectation: The result is correct
    """
    compiled_op = torch.compile(setitem_slice_dynamic_partial_op, backend=backend)

    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    cpu_value = np.random.uniform(0.5, 2, shape).astype(dtype)

    npu_input = torch.from_numpy(cpu_input).npu()
    npu_value = torch.from_numpy(cpu_value).npu()

    cpu_output = setitem_slice_dynamic_partial_op(
        torch.from_numpy(cpu_input), torch.from_numpy(cpu_value)).detach().numpy()
    npu_output = compiled_op(npu_input, npu_value).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output)


# ============================================================================
# int32 tests
# ============================================================================

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_strided_slice_assign_int32():
    """
    Feature: Test aclnn strided_slice_assign with int32
    Description: Test slice assignment with int32 dtype
    Expectation: The result is correct
    """
    shape = (8, 16)

    def setitem_int_op(x, value):
        x = x.clone()
        x[1:5] = value
        return x

    compiled_op = torch.compile(setitem_int_op, backend=backend)

    cpu_input = np.random.randint(-100, 100, shape).astype(np.int32)
    cpu_value = np.random.randint(-50, 50, (4, shape[1])).astype(np.int32)

    npu_input = torch.from_numpy(cpu_input).npu()
    npu_value = torch.from_numpy(cpu_value).npu()

    cpu_output = setitem_int_op(torch.from_numpy(cpu_input), torch.from_numpy(cpu_value)).detach().numpy()
    npu_output = compiled_op(npu_input, npu_value).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output, 0)
