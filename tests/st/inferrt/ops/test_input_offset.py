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


def _assert_empty_result_matches(eager_result, compiled_result):
    """Check empty tensor shape and values against eager mode."""
    assert eager_result.numel() == 0
    assert compiled_result.numel() == 0
    assert tuple(compiled_result.shape) == tuple(eager_result.shape)
    assert tuple(compiled_result.stride()) == tuple(eager_result.stride())
    torch.testing.assert_close(compiled_result.detach().cpu(), eager_result.detach().cpu())


def _empty_as_strided_input(dtype, shape, stride, storage_offset):
    """Create an empty NPU tensor with a specific storage offset."""
    base = torch.empty(4, dtype=dtype).npu()
    return torch.as_strided(base, shape, stride, storage_offset=storage_offset)


def _compile_and_run_empty_func(func, *args):
    """Compile and run an empty tensor case without reusing cached empty output storage."""
    torch.compiler.reset()
    compiled_func = torch.compile(func, backend=backend)
    return compiled_func(*args)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0, 3)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_empty_input_without_storage_for_unary_op(shape, dtype):
    """
    Feature: Test empty tensor input without storage offset
    Description: Test pybind runtime input update when empty tensor storage bytes and offset bytes are both zero
    Expectation: Empty input runs without storage-offset bounds failure and matches eager mode
    """
    x = torch.empty(shape, dtype=dtype).npu()
    assert x.numel() == 0
    assert x.storage_offset() == 0

    result_eager = sigmoid_func(x)
    result_compiled = _compile_and_run_empty_func(sigmoid_func, x)

    _assert_empty_result_matches(result_eager, result_compiled)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "storage_offset",
    [1, 4, 2031616],
    ids=[
        "offset_bytes_lt_storage_bytes",
        "offset_bytes_eq_storage_bytes",
        "offset_bytes_gt_storage_bytes",
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_empty_input_offset_bounds_for_unary_op(storage_offset, dtype):
    """
    Feature: Test empty tensor input with storage offset
    Description: Test pybind runtime input update when empty tensor offset bytes are below/equal/above storage bytes
    Expectation: Empty input runs without storage-offset bounds failure and matches eager mode
    """
    x = _empty_as_strided_input(dtype, (0,), (1,), storage_offset)
    assert x.numel() == 0

    result_eager = sigmoid_func(x)
    result_compiled = _compile_and_run_empty_func(sigmoid_func, x)

    _assert_empty_result_matches(result_eager, result_compiled)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "shape,stride,storage_offset",
    [
        ((0, 3), (3, 1), 1),
        ((0, 3), (3, 1), 4),
        ((0, 3), (3, 1), 2031616),
        ((2, 0, 3), (3, 3, 1), 2031616),
    ],
    ids=[
        "2d_offset_bytes_lt_storage_bytes",
        "2d_offset_bytes_eq_storage_bytes",
        "2d_offset_bytes_gt_storage_bytes",
        "3d_offset_bytes_gt_storage_bytes",
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_empty_input_offset_bounds_for_ranked_unary_op(shape, stride, storage_offset, dtype):
    """
    Feature: Test ranked empty tensor input with storage offset
    Description: Test empty tensors with rank > 1 and large storage offsets through pybind input update
    Expectation: Compiled result matches eager mode without storage-offset bounds failure
    """
    x = _empty_as_strided_input(dtype, shape, stride, storage_offset)
    assert x.numel() == 0

    result_eager = sigmoid_func(x)
    result_compiled = _compile_and_run_empty_func(sigmoid_func, x)

    _assert_empty_result_matches(result_eager, result_compiled)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_empty_input_offset_bounds_for_binary_op(dtype):
    """
    Feature: Test binary operation with empty storage-offset inputs
    Description: Test two empty inputs whose storage offset bytes equal and exceed their storage bytes
    Expectation: Compiled result matches eager mode without storage-offset bounds failure
    """
    x = _empty_as_strided_input(dtype, (0,), (1,), 4)
    y = _empty_as_strided_input(dtype, (0,), (1,), 2031616)

    result_eager = add_func(x, y)
    result_compiled = _compile_and_run_empty_func(add_func, x, y)

    _assert_empty_result_matches(result_eager, result_compiled)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_non_empty_input_offset_out_of_bounds_rejected_by_torch_npu(dtype):
    """
    Feature: Test non-empty tensor input with out-of-bounds storage offset
    Description: Verify torch_npu rejects invalid non-empty storage offsets before InferRT input update
    Expectation: torch_npu raises an out-of-bounds error
    """
    base = torch.empty(4, dtype=dtype).npu()
    with pytest.raises(RuntimeError, match="out of bounds"):
        torch.as_strided(base, (1,), (1,), storage_offset=2031616)
    torch.npu.synchronize()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
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
