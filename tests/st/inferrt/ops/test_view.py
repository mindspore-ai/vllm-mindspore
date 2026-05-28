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
import re

import numpy as np
import pytest
import torch
import torch_npu

from torch_npu.testing.common_utils import create_common_tensor

from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual

_EAGER_VIEW_ERR = r"view size is not compatible"
_DYNAMO_VIEW_ERR = r"Cannot view a tensor with shape"


def build_non_contiguous_tensor(input_self_tensor, pattern):
    """Build a deterministic non-contiguous view tensor from a base input tensor."""
    if pattern == "permute_0132":
        return input_self_tensor.permute(0, 1, 3, 2)
    if pattern == "transpose_23":
        return input_self_tensor.transpose(2, 3)
    if pattern == "last_dim_narrow":
        return input_self_tensor[..., 1:]
    if pattern == "dim1_narrow":
        return input_self_tensor[:, 1:, ...]
    if pattern == "dim2_narrow":
        return input_self_tensor[:, :, 1:, ...]
    if pattern == "dim1_narrow_transpose_01":
        return input_self_tensor[:, 1:, ...].transpose(0, 1)
    if pattern == "transpose_01":
        return input_self_tensor.transpose(0, 1)
    raise ValueError(f"unsupported pattern: {pattern}")


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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "api,input_shape,target_shape,input_format,error_pattern",
    [
        ("view", (2, 3, 4, 5), (2, 3, 1, 4, 5), 3, r"does not support view metadata change"),
        ("view", (2, 3, 4, 5, 6), (2, 3, 4, 30), 32, r"does not support view metadata change"),
        ("reshape", (2, 3, 4, 5), (2, 3, 1, 4, 5), 3, r"does not support view metadata change"),
        ("reshape", (2, 3, 4, 5, 6), (2, 3, 4, 30), 32, r"does not support view metadata change"),
    ],
    ids=[
        "view_nc1hwc0_base_nchw_to_rank5_reject",
        "view_ndc1hwc0_base_ncdhw_to_non_rank5_reject",
        "reshape_nc1hwc0_base_nchw_to_rank5_reject",
        "reshape_ndc1hwc0_base_ncdhw_to_non_rank5_reject",
    ],
)
def test_view_like_format_metadata_change_error(api, input_shape, target_shape, input_format, error_pattern):
    """
    Feature: View-like metadata-change format check
    Description: Reject view/reshape metadata changes that require implicit base format switching.
    Expectation: Compiled execution raises RuntimeError with explicit message.
    """

    def func(x: torch.Tensor) -> torch.Tensor:
        if api == "view":
            return x.view(target_shape)
        return x.reshape(target_shape)

    x = torch.randn(*input_shape, dtype=torch.float16).npu()
    x = torch_npu.npu_format_cast(x.contiguous(), input_format)
    compiled_func = torch.compile(func, backend=backend)
    with pytest.raises(RuntimeError, match=error_pattern):
        compiled_func(x)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "input_shape,target_shape,input_format",
    [
        ((2, 3, 4, 5), (2, 3, 20), 3),
        ((2, 3, 4, 5, 6), (2, 3, 2, 10, 6), 32),
    ],
    ids=["nc1hwc0_rank4_to_rank3_allow", "ndc1hwc0_rank5_to_rank5_allow"],
)
def test_view_format_metadata_change_allowed(input_shape, target_shape, input_format):
    """
    Feature: View metadata-change format check
    Description: Allow view shape changes that do not require implicit base format switching.
    Expectation: Compiled execution is not rejected by CheckViewMetaDataChangeForFormat.
    """

    def func(x: torch.Tensor) -> torch.Tensor:
        return x.view(target_shape)

    x = torch.randn(*input_shape, dtype=torch.float16).npu()
    x = torch_npu.npu_format_cast(x.contiguous(), input_format)
    compiled_func = torch.compile(func, backend=backend)

    eager_out = func(x)
    assert tuple(eager_out.shape) == tuple(target_shape)
    with pytest.raises(RuntimeError, match="Network output does not support non-base memory format"):
        compiled_func(x)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_view_permute_view_input():
    """
    Feature: Test view with non-contiguous (permuted) input
    Description: View on permute(0,1,3,2) tensor where only the first two dims are viewed
    Expectation: The result is correct
    """
    # permute(0,1,3,2) on [3,4,5,6] -> [3,4,6,5], strides (120,30,1,6)
    # dims 0,1 are contiguous: strides[0]=120 == shape[1]*strides[1]=4*30=120
    # view(12, 6, 5): merges dims [0,1] which are contiguous, keeps dims [2,3] unchanged
    self_tensor = torch.rand(3, 4, 5, 6, dtype=torch.bfloat16)
    permuted = self_tensor.permute(0, 1, 3, 2)
    cpu_output = permuted.view(12, 6, 5)

    def func(x):
        return x.permute(0, 1, 3, 2).view(12, 6, 5)

    compiled_func = torch.compile(func, backend=backend)
    npu_output = compiled_func(self_tensor.npu())
    npu_output_cpu = npu_output.detach().cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_view_slice_offset_input():
    """
    Feature: Test view with sliced input (non-zero storage offset)
    Description: View on sliced tensor with storage_offset != 0 but contiguous strides
    Expectation: The result is correct
    """
    self_tensor = torch.rand(3, 4, 5, 6, dtype=torch.bfloat16)
    sliced = self_tensor[1:]  # shape [2,4,5,6], contiguous strides, storage_offset != 0

    cpu_output = sliced.view(2, 4, 30)

    def func(x):
        return x[1:].view(2, 4, 30)

    compiled_func = torch.compile(func, backend=backend)
    npu_output = compiled_func(self_tensor.npu())
    npu_output_cpu = npu_output.detach().cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "shape, slice_pattern, target_shape",
    [
        ((3, 4, 5, 6), "last_dim_narrow", (3, 4, 5, 5)),
        ((4, 5, 6, 7), "dim1_narrow", (4, 4, 42)),
        ((2, 3, 4, 5, 6), "dim2_narrow", (2, 3, 3, 30)),
        ((4, 5, 6, 7), "transpose_01", (5, 4, 42)),
        ((3, 4, 5, 6), "permute_0132", (12, 6, 5)),
        ((3, 4, 5, 6), "dim1_narrow_transpose_01", (3, 3, 30)),
        ((3, 4, 5, 6), "dim2_narrow", (3, 4, 4, 6)),
    ],
)
def test_view_noncontiguous_view_compatible(shape, slice_pattern, target_shape):
    """
    Feature: view on non-contiguous input
    Description: Non-contiguous tensors can still be viewed when stride geometry is compatible
    Expectation: Compiled result equals eager result
    """
    self_tensor = torch.rand(shape, dtype=torch.bfloat16)
    self_tensor_npu = self_tensor.npu()
    non_contiguous = build_non_contiguous_tensor(self_tensor, slice_pattern)
    assert not non_contiguous.is_contiguous()

    def custom_op_func(input_self_tensor):
        inter = build_non_contiguous_tensor(input_self_tensor, slice_pattern)
        return inter.view(target_shape)

    cpu_output = custom_op_func(self_tensor)

    op_func_compiled = torch.compile(custom_op_func, backend=backend)
    npu_output = op_func_compiled(self_tensor_npu)
    npu_output_cpu = npu_output.detach().cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "shape, transform_pattern, target_shape",
    [
        ((3, 4, 5, 6), "permute_0132", (3, 4, 30)),
        ((4, 5, 6, 7), "transpose_01", (20, 42)),
    ],
)
def test_view_noncontiguous_view_incompatible(shape, transform_pattern, target_shape):
    """
    Feature: view on non-contiguous input with incompatible stride geometry
    Description: Non-contiguous intermediate whose target shape cannot infer legal strides,
                 so view should fail (PyTorch eager also rejects this view).
    Expectation: Dynamo FakeTensor tracing raises RuntimeError (view on non-contiguous
                 tensor is rejected during abstract interpretation before reaching InferRT).
    """
    self_tensor = torch.rand(shape, dtype=torch.bfloat16)
    self_tensor_npu = self_tensor.npu()

    transformed = build_non_contiguous_tensor(self_tensor, transform_pattern)
    assert not transformed.is_contiguous()

    # Verify PyTorch eager also rejects this view
    with pytest.raises(RuntimeError, match=_EAGER_VIEW_ERR):
        transformed.view(target_shape)

    def custom_op_func(input_self_tensor):
        inter = build_non_contiguous_tensor(input_self_tensor, transform_pattern)
        return inter.view(target_shape)

    op_func_compiled = torch.compile(custom_op_func, backend=backend, fullgraph=True)
    # Dynamo FakeTensor rejects non-contiguous view during tracing
    with pytest.raises(RuntimeError, match=_DYNAMO_VIEW_ERR):
        op_func_compiled(self_tensor_npu)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "input_shape,target_shape,error_substr",
    [
        ((2, 3, 4), (-1, -1), "only one dimension can be inferred"),
        ((2, 3, 4), (2, -2, 6), "invalid shape dimension"),
        ((2, 3, 4), (5, 5), "is invalid for input of size 24"),
        ((2, 3, 4), (5, -1), "is invalid for input of size 24"),
        ((0, 3, 4), (-1, 0), "tensor of 0 elements"),
        ((0, 3, 4), (0, -1), "tensor of 0 elements"),
    ],
    ids=[
        "multiple_minus_one",
        "dimension_less_than_minus_one",
        "numel_mismatch_without_minus_one",
        "numel_mismatch_with_minus_one",
        "zero_numel_ambiguous_minus_one_zero",
        "zero_numel_ambiguous_zero_minus_one",
    ],
)
def test_view_infer_size_invalid_cases_match_eager(input_shape, target_shape, error_substr):
    """
    Feature: view infer-size validation
    Description: Verify InferRT reuses torch/FakeTensor validation for invalid view shapes.
    Expectation: Eager and compiled execution both fail before runtime view metadata handling.
    """

    def func(x: torch.Tensor) -> torch.Tensor:
        return x.view(target_shape)

    cpu_input = torch.randn(*input_shape, dtype=torch.float16)
    npu_input = cpu_input.npu()

    with pytest.raises(RuntimeError, match=re.escape(error_substr)):
        func(cpu_input)

    compiled_func = torch.compile(func, backend=backend, fullgraph=True)
    with pytest.raises(RuntimeError, match=re.escape(error_substr)):
        compiled_func(npu_input)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "input_shape,target_shape",
    [
        ((2, 3, 4), (2, -1, 2)),
        ((0, 3, 4), (0, 12)),
        ((0,), (-1,)),
    ],
    ids=["infer_minus_one_valid", "zero_numel_explicit_shape", "zero_numel_single_minus_one"],
)
def test_view_infer_size_valid_cases_match_eager(input_shape, target_shape):
    """
    Feature: view infer-size validation
    Description: Verify valid shapes inferred by torch are consumed correctly by InferRT view.
    Expectation: Compiled and eager outputs have identical shape, stride, and values.
    """

    def func(x: torch.Tensor) -> torch.Tensor:
        return x.view(target_shape)

    cpu_input = torch.randn(*input_shape, dtype=torch.float16)
    npu_input = cpu_input.npu()

    eager_out = func(cpu_input)
    compiled_func = torch.compile(func, backend=backend, fullgraph=True)
    compiled_out = compiled_func(npu_input).cpu()

    assert tuple(compiled_out.shape) == tuple(eager_out.shape)
    assert compiled_out.stride() == eager_out.stride()
    AssertRtolEqual(eager_out.numpy(), compiled_out.numpy())
