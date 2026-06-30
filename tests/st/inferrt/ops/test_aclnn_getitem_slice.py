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

"""Tests for aclnn getitem_slice operation (tensor slicing)."""
import numpy as np
import pytest
import torch

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


# ============================================================================
# Basic slice op functions
# ============================================================================

def slice_basic_op(x):
    """Basic slice: x[1:5]"""
    return x[1:5]


def slice_start_only_op(x):
    """Slice with start only: x[3:]"""
    return x[3:]


def slice_end_only_op(x):
    """Slice with end only: x[:5]"""
    return x[:5]


def slice_with_step_op(x):
    """Slice with step: x[0:10:2]"""
    return x[0:10:2]


def slice_negative_indices_op(x):
    """Slice with negative indices: x[-5:-1]"""
    return x[-5:-1]


def slice_negative_step_op(x):
    """Slice with negative step: x[10:0:-2]"""
    return x[10:0:-2]


def slice_multi_dim_op(x):
    """Multi-dimensional slice: x[1:3, 2:6]"""
    return x[1:3, 2:6]


def slice_3d_op(x):
    """3D slice: x[0:2, 1:4, 3:7]"""
    return x[0:2, 1:4, 3:7]


def slice_4d_op(x):
    """4D slice: x[0:1, :, 2:5, :]"""
    return x[0:1, :, 2:5, :]


def slice_full_dim_op(x):
    """Slice entire dimension with step: x[:, ::2]"""
    return x[:, ::2]


# ============================================================================
# Dynamic shape slice op functions
# ============================================================================

def slice_dynamic_op(x):
    """Dynamic slice: x[:b] where b = x.size(0)"""
    b = x.size(0)
    return x[:b]


def slice_dynamic_2d_op(x):
    """Dynamic 2D slice: x[:b, :c]"""
    b = x.size(0)
    c = x.size(1)
    return x[:b, :c]


def slice_dynamic_multi_dim_op(x):
    """Dynamic multi-dim slice: x[:b, 2:c]"""
    b = x.size(0)
    c = x.size(1)
    return x[:b, 2:c]


# ============================================================================
# Helper
# ============================================================================

def _get_prec(dtype):
    """Get precision tolerance based on dtype."""
    if dtype in (np.float16, np.bfloat16):
        return 0.001
    return 0.0001


def _to_np_dtype(torch_dtype):
    """Convert torch dtype to numpy dtype."""
    mapping = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.bfloat16: np.float16,  # numpy doesn't have bfloat16, use float16 for prec
        torch.int32: np.int32,
    }
    return mapping.get(torch_dtype, np.float32)


# ============================================================================
# Static shape tests
# ============================================================================

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(16,), (32,), (64,)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_getitem_slice_basic(shape, dtype):
    """
    Feature: Test aclnn getitem_slice
    Description: Test basic slicing x[1:5] with 1D tensors
    Expectation: The result is correct
    """
    compiled_op = torch.compile(slice_basic_op, backend=backend)
    prec = _get_prec(dtype)

    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    npu_input = torch.from_numpy(cpu_input).npu()

    cpu_output = slice_basic_op(torch.from_numpy(cpu_input)).detach().numpy()
    npu_output = compiled_op(npu_input).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output, prec)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8, 16), (16, 32), (64, 128)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_getitem_slice_2d_basic(shape, dtype):
    """
    Feature: Test aclnn getitem_slice on 2D tensors
    Description: Test x[1:5] on 2D tensors
    Expectation: The result is correct
    """
    compiled_op = torch.compile(slice_basic_op, backend=backend)
    prec = _get_prec(dtype)

    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    npu_input = torch.from_numpy(cpu_input).npu()

    cpu_output = slice_basic_op(torch.from_numpy(cpu_input)).detach().numpy()
    npu_output = compiled_op(npu_input).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output, prec)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8, 16), (16, 32)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_getitem_slice_with_step(shape, dtype):
    """
    Feature: Test aclnn getitem_slice with step
    Description: Test x[0:10:2] slicing with step on 2D tensors
    Expectation: The result is correct
    """
    compiled_op = torch.compile(slice_with_step_op, backend=backend)
    prec = _get_prec(dtype)

    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    npu_input = torch.from_numpy(cpu_input).npu()

    cpu_output = slice_with_step_op(torch.from_numpy(cpu_input)).detach().numpy()
    npu_output = compiled_op(npu_input).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output, prec)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8, 16), (16, 32)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_getitem_slice_negative_indices(shape, dtype):
    """
    Feature: Test aclnn getitem_slice with negative indices
    Description: Test x[-5:-1] slicing with negative indices
    Expectation: The result is correct
    """
    compiled_op = torch.compile(slice_negative_indices_op, backend=backend)
    prec = _get_prec(dtype)

    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    npu_input = torch.from_numpy(cpu_input).npu()

    cpu_output = slice_negative_indices_op(torch.from_numpy(cpu_input)).detach().numpy()
    npu_output = compiled_op(npu_input).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output, prec)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_getitem_slice_multi_dim(dtype):
    """
    Feature: Test aclnn getitem_slice on multi-dimensional tensors
    Description: Test x[1:3, 2:6] on 2D tensors with multi-dim slice
    Expectation: The result is correct
    """
    shape = (8, 16)
    compiled_op = torch.compile(slice_multi_dim_op, backend=backend)
    prec = _get_prec(dtype)

    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    npu_input = torch.from_numpy(cpu_input).npu()

    cpu_output = slice_multi_dim_op(torch.from_numpy(cpu_input)).detach().numpy()
    npu_output = compiled_op(npu_input).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output, prec)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_getitem_slice_3d(dtype):
    """
    Feature: Test aclnn getitem_slice on 3D tensors
    Description: Test x[0:2, 1:4, 3:7] on 3D tensors
    Expectation: The result is correct
    """
    shape = (4, 8, 16)
    compiled_op = torch.compile(slice_3d_op, backend=backend)
    prec = _get_prec(dtype)

    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    npu_input = torch.from_numpy(cpu_input).npu()

    cpu_output = slice_3d_op(torch.from_numpy(cpu_input)).detach().numpy()
    npu_output = compiled_op(npu_input).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output, prec)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_getitem_slice_4d(dtype):
    """
    Feature: Test aclnn getitem_slice on 4D tensors
    Description: Test x[0:1, :, 2:5, :] on 4D tensors
    Expectation: The result is correct
    """
    shape = (2, 4, 8, 16)
    compiled_op = torch.compile(slice_4d_op, backend=backend)
    prec = _get_prec(dtype)

    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    npu_input = torch.from_numpy(cpu_input).npu()

    cpu_output = slice_4d_op(torch.from_numpy(cpu_input)).detach().numpy()
    npu_output = compiled_op(npu_input).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output, prec)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_getitem_slice_full_dim_with_step(dtype):
    """
    Feature: Test aclnn getitem_slice with full dimension and step
    Description: Test x[:, ::2] slicing entire dim with step
    Expectation: The result is correct
    """
    shape = (8, 16)
    compiled_op = torch.compile(slice_full_dim_op, backend=backend)
    prec = _get_prec(dtype)

    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    npu_input = torch.from_numpy(cpu_input).npu()

    cpu_output = slice_full_dim_op(torch.from_numpy(cpu_input)).detach().numpy()
    npu_output = compiled_op(npu_input).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output, prec)


# ============================================================================
# Dynamic shape tests
# ============================================================================

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8,), (4, 8), (16, 32), (64, 128), (2, 4, 8)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_getitem_slice_dynamic(shape, dtype):
    """
    Feature: Test aclnn getitem_slice with dynamic shape
    Description: Test x[:b] dynamic slicing pattern
    Expectation: The result is correct
    """
    compiled_op = torch.compile(slice_dynamic_op, backend=backend)
    prec = _get_prec(dtype)

    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    npu_input = torch.from_numpy(cpu_input).npu()

    cpu_output = slice_dynamic_op(torch.from_numpy(cpu_input)).detach().numpy()
    npu_output = compiled_op(npu_input).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output, prec)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(4, 8), (16, 32), (64, 128)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_getitem_slice_dynamic_2d(shape, dtype):
    """
    Feature: Test aclnn getitem_slice with 2D dynamic shape
    Description: Test x[:b, :c] dynamic slicing on 2D tensors
    Expectation: The result is correct
    """
    compiled_op = torch.compile(slice_dynamic_2d_op, backend=backend)
    prec = _get_prec(dtype)

    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    npu_input = torch.from_numpy(cpu_input).npu()

    cpu_output = slice_dynamic_2d_op(torch.from_numpy(cpu_input)).detach().numpy()
    npu_output = compiled_op(npu_input).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output, prec)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(4, 8), (16, 32)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_getitem_slice_dynamic_multi_dim(shape, dtype):
    """
    Feature: Test aclnn getitem_slice with multi-dim dynamic shape
    Description: Test x[:b, 2:c] dynamic slicing with partial static dims
    Expectation: The result is correct
    """
    compiled_op = torch.compile(slice_dynamic_multi_dim_op, backend=backend)
    prec = _get_prec(dtype)

    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    npu_input = torch.from_numpy(cpu_input).npu()

    cpu_output = slice_dynamic_multi_dim_op(torch.from_numpy(cpu_input)).detach().numpy()
    npu_output = compiled_op(npu_input).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output, prec)


# ============================================================================
# bfloat16 tests
# ============================================================================

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8, 16), (16, 32)])
def test_getitem_slice_bfloat16(shape):
    """
    Feature: Test aclnn getitem_slice with bfloat16
    Description: Test basic slicing with bf16 dtype
    Expectation: The result is correct
    """
    dtype = torch.bfloat16

    def slice_bf16_op(x):
        return x[1:5]

    compiled_op = torch.compile(slice_bf16_op, backend=backend)

    npu_input = torch.randn(shape, dtype=dtype).npu()
    # Eager reference on NPU
    expected = slice_bf16_op(npu_input.clone())

    npu_output = compiled_op(npu_input)
    AssertRtolEqual(expected.detach().cpu(), npu_output.detach().cpu(), 0.001)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_getitem_slice_int32():
    """
    Feature: Test aclnn getitem_slice with int32
    Description: Test slicing with int32 dtype
    Expectation: The result is correct
    """
    shape = (8, 16)

    def slice_int_op(x):
        return x[1:5]

    compiled_op = torch.compile(slice_int_op, backend=backend)

    cpu_input = np.random.randint(-100, 100, shape).astype(np.int32)
    npu_input = torch.from_numpy(cpu_input).npu()

    cpu_output = slice_int_op(torch.from_numpy(cpu_input)).detach().numpy()
    npu_output = compiled_op(npu_input).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output, 0)
