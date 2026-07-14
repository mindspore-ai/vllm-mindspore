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
"""Tests for aten.unflatten.int lowering to InferRT unflatten_view."""

import pytest
import torch

from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark


def _assert_view_metadata_matches(eager_out, compiled_out):
    assert tuple(compiled_out.shape) == tuple(eager_out.shape)
    assert tuple(compiled_out.stride()) == tuple(eager_out.stride())
    assert compiled_out.storage_offset() == eager_out.storage_offset()
    assert compiled_out.is_contiguous() == eager_out.is_contiguous()


def _assert_unflatten_matches_eager(func, input_tensor):
    eager_out = func(input_tensor)
    compiled_func = torch.compile(func, backend=backend, fullgraph=True)
    compiled_out = compiled_func(input_tensor)

    _assert_view_metadata_matches(eager_out, compiled_out)
    torch.testing.assert_close(compiled_out, eager_out)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "shape, dim, sizes, dtype",
    [
        ((0,), 0, (0, 1), torch.float32),
        ((4,), 0, [2, 2], torch.int32),
        ((4,), 0, torch.Size([2, 2]), torch.float32),
        ((4, 12), 1, (2, -1, 3), torch.float32),
        ((2, 3, 20), -1, (4, 5), torch.bfloat16),
        ((24,), 0, (2, 3, 4), torch.int32),
        ((2, 10), 1, (-1,), torch.bfloat16),
        ((2, 360), 1, (3, 4, -1, 6), torch.float32),
        ((2, 0, 2), 1, (3, -1, 4, 5), torch.float32),
    ],
)
def test_unflatten_method_matches_eager(shape, dim, sizes, dtype):
    """
    Feature: Tensor.unflatten lowering
    Description: Verify method unflatten lowers to InferRT unflatten_view and matches eager view metadata.
    Expectation: Compiled output matches eager output.
    """
    input_tensor = torch.arange(int(torch.prod(torch.tensor(shape))), dtype=dtype).reshape(shape).npu()

    def func(x):
        return x.unflatten(dim, sizes)

    _assert_unflatten_matches_eager(func, input_tensor)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_aten_unflatten_int_matches_eager():
    """
    Feature: aten.unflatten.int lowering
    Description: Verify explicit aten.unflatten.int target maps to InferRT unflatten_view.
    Expectation: Compiled output matches eager output.
    """
    input_tensor = torch.arange(48, dtype=torch.float32).reshape(2, 24).npu()

    def func(x):
        return torch.ops.aten.unflatten.int(x, 1, [3, -1, 4])

    _assert_unflatten_matches_eager(func, input_tensor)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "transform, dim, sizes",
    [
        ("transpose_01", 2, (2, 2)),
        ("permute_021", 1, (2, 2)),
        ("strided_slice", 1, (1, 3)),
    ],
)
def test_unflatten_non_contiguous_input_matches_eager(transform, dim, sizes):
    """
    Feature: unflatten on non-contiguous inputs
    Description: Verify InferRT preserves torch view metadata for view-compatible non-contiguous inputs.
    Expectation: Compiled output shape, stride, storage offset, and values match eager mode.
    """
    base = torch.arange(48, dtype=torch.float32).reshape(2, 6, 4).npu()
    if transform == "transpose_01":
        input_tensor = base[:, :3, :].transpose(0, 1)
    elif transform == "permute_021":
        input_tensor = base[:, :3, :].permute(0, 2, 1)
    elif transform == "strided_slice":
        input_tensor = base[:, ::2, :]
    else:
        raise ValueError(f"Unsupported transform: {transform}")
    assert not input_tensor.is_contiguous()

    def func(x):
        return x.unflatten(dim, sizes)

    _assert_unflatten_matches_eager(func, input_tensor)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_unflatten_zero_size_dim_matches_eager():
    """
    Feature: unflatten with zero-size dimensions
    Description: Verify torch-inferred zero-size output shape is preserved by InferRT unflatten_view.
    Expectation: Compiled output matches eager output.
    """
    input_tensor = torch.empty((2, 0, 4), dtype=torch.float32, device="npu")

    def func(x):
        return x.unflatten(1, (0, 3))

    _assert_unflatten_matches_eager(func, input_tensor)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_unflatten_output_consumed_by_add_matches_eager():
    """
    Feature: unflatten output consumed by following op
    Description: Verify downstream ops consume InferRT unflatten_view output with the expected shape and values.
    Expectation: Compiled output matches eager output.
    """
    input_tensor = torch.arange(24, dtype=torch.float32).reshape(2, 12).npu()

    def func(x):
        return x.unflatten(1, (3, 4)) + 1.5

    eager_out = func(input_tensor)
    compiled_func = torch.compile(func, backend=backend, fullgraph=True)
    compiled_out = compiled_func(input_tensor)
    torch.testing.assert_close(compiled_out, eager_out)
