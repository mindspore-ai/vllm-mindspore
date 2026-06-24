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
"""Tests for torch.reshape operation."""
import numpy as np
import pytest
import torch

from torch_npu.testing.common_utils import create_common_tensor

from ms_inferrt.torch import backend as fx_backend
from ms_inferrt.torch import fx_mlir_backend as backend
from ms_inferrt.torch.fx_backend import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual

_VIEW_ERR = r"View encountered unsupported non-contiguous input tensor"


# pylint: disable=redefined-builtin
def op_func(input, shape):
    """op function for reshape"""
    return input.reshape(shape)


def reshape_forward(shape_format, op_func_compiled):
    """
    reshape forward function
    Args:
        shape_format: list of [dtype, format, shape]
        op_func_compiled: The compiled op function.
    """
    for item in shape_format:
        cpu_input, npu_input = create_common_tensor(item, 0, 100)
        shape = [4, 16]
        cpu_output = op_func(cpu_input, shape).detach().numpy()
        npu_output = op_func_compiled(npu_input, shape).detach().cpu().numpy()
        AssertRtolEqual(cpu_output, npu_output)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_reshape():
    """
    Feature: Test reshape
    Description: Test reshape op with mlir_backend
    Expectation: The result is correct
    """
    dtype_list = [np.float16, np.float32, np.int32, np.bool_]
    format_list = [0]
    shape_list = [[8, 8], [2, 4, 8], [2, 4, 4, 2]]

    shape_format = [
        [i, j, k] for i in dtype_list for j in format_list for k in shape_list
    ]
    op_func_compiled = torch.compile(op_func, backend=backend)
    reshape_forward(shape_format, op_func_compiled)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_aten_reshape_default_to_view():
    """
    Feature: Test aten.reshape.default lowering
    Description: Verify aten.reshape.default is lowered to InferRT view op through fx_backend
    Expectation: The result is correct
    """
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).npu()

    def func(input_tensor):
        return torch.ops.aten.reshape.default(input_tensor, [6, 4])

    eager_out = func(x)
    compiled_func = torch.compile(func, backend=fx_backend, fullgraph=True)
    compiled_out = compiled_func(x)
    torch.testing.assert_close(compiled_out, eager_out)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "shape, transform_pattern, target_shape",
    [
        ((3, 4, 5, 6), "permute_0132", (360,)),
        ((3, 4, 5, 6), "permute_0132", (3, 4, 30)),
        ((4, 5, 6, 7), "transpose_01", (20, 42)),
    ],
)
def test_reshape_noncontiguous_view_incompatible(shape, transform_pattern, target_shape):
    """
    Feature: reshape on non-contiguous input with incompatible stride geometry
    Description: Unlike view, reshape succeeds in eager mode (contiguous fallback), but InferRT's
                 reshape (implemented via view.cc) rejects non-view-compatible stride geometry.
                 Dynamo FakeTensor does NOT intercept reshape (eager always succeeds), so the
                 error comes from InferRT at runtime.
    Expectation: InferRT raises RuntimeError for unsupported non-contiguous input.
    """
    self_tensor = torch.rand(shape, dtype=torch.bfloat16)

    # Build non-contiguous tensor
    if transform_pattern == "permute_0132":
        transformed = self_tensor.permute(0, 1, 3, 2)
    elif transform_pattern == "transpose_01":
        transformed = self_tensor.transpose(0, 1)
    else:
        raise ValueError(f"unsupported transform pattern: {transform_pattern}")
    assert not transformed.is_contiguous()

    # Eager reshape succeeds (contiguous fallback)
    eager_result = transformed.reshape(target_shape)
    assert eager_result.data_ptr() != transformed.data_ptr()

    # Pass non-contiguous tensor directly as input to compiled func
    transformed_npu = transformed.npu()

    def func(x):
        return x.reshape(target_shape)

    op_func_compiled = torch.compile(func, backend=backend, fullgraph=True)
    with pytest.raises(RuntimeError, match=_VIEW_ERR):
        op_func_compiled(transformed_npu)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_aten_reshape():
    """
    Feature: Test aten.reshape via fx_backend
    Description: Test aten.reshape interface (torch.ops.aten.reshape.default) with fx_backend backend
    Expectation: The result matches torch reference
    """
    def aten_reshape_func(x, shape):
        return torch.ops.aten.reshape.default(x, shape)

    op_func_compiled = torch.compile(aten_reshape_func, backend=fx_backend)
    x = torch.randn(2, 3, 4, dtype=torch.float32).npu()
    shape = (6, 4)
    npu_out = op_func_compiled(x, shape)
    cpu_out = torch.ops.aten.reshape.default(x.cpu(), shape)
    AssertRtolEqual(cpu_out, npu_out.detach().cpu())
    assert npu_out.device.type == "npu"
