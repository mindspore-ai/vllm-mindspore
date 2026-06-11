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
"""Tests for NZ view outputs consumed by NZ-capable aclnn operators."""

import pytest
import torch
import torch._dynamo.config as dynamo_config
import torch.nn.functional as F
import torch_npu

from ms_inferrt.torch.fx_backend import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


NZ_FORMAT = 29
LINEAR_VIEW_CASES = [
    ("alias", (8, 16)),
    ("permute", (8, 16)),
    ("view", (1, 8, 16)),
    ("reshape", (1, 8, 16)),
    ("flatten", (1, 8, 16)),
    ("slice", (12, 16)),
    ("getitem_slice", (12, 16)),
    ("select", (1, 8, 16)),
    ("select_method", (1, 8, 16)),
    ("narrow", (12, 16)),
    ("squeeze", (1, 8, 16)),
    ("unsqueeze", (8, 16)),
    ("split_tensor", (16, 16)),
    ("split_with_size", (16, 16)),
    ("chunk", (16, 16)),
    ("unbind", (1, 8, 16)),
]
QUANT_MATMUL_SHAPES = [(8, 16, 32), (4, 32, 48), (3, 16, 64)]
QUANT_MATMUL_VIEW_CASES = ["permute", "view", "reshape", "flatten", "squeeze", "select", "select_method", "unbind"]

dynamo_config.cache_size_limit = 64


def _make_nz_tensor(shape, dtype=torch.float16):
    """Create an NPU tensor in FRACTAL_NZ format."""
    torch_npu.npu.config.allow_internal_format = True
    x = torch.randn(*shape, dtype=dtype).npu()
    return torch_npu.npu_format_cast(x.contiguous(), NZ_FORMAT)


def _make_linear_nz_view_input(case, shape):
    """Build an NZ input whose graph view output is consumed by linear."""
    x = _make_nz_tensor(shape)
    if case == "permute":
        return x.t()
    return x


def _apply_linear_nz_view(input_tensor, case):
    """Apply a view op whose output is consumed by linear."""
    if case == "alias":
        return torch.ops.aten.alias.default(input_tensor)
    if case == "permute":
        return input_tensor.t()
    if case == "view":
        return input_tensor.view(8, 16)
    if case == "reshape":
        return input_tensor.reshape(8, 16)
    if case == "flatten":
        return input_tensor.flatten(0, 1)
    if case == "slice":
        return torch.ops.aten.slice.Tensor(input_tensor, 0, 2, 10, 1)
    if case == "getitem_slice":
        return input_tensor[2:10, :]
    if case == "select":
        return torch.select(input_tensor, 0, 0)
    if case == "select_method":
        return input_tensor.select(0, 0)
    if case == "narrow":
        return torch.narrow(input_tensor, 0, 2, 8)
    if case == "squeeze":
        return input_tensor.squeeze(0)
    if case == "unsqueeze":
        return input_tensor.unsqueeze(0)
    if case == "split_tensor":
        return torch.split(input_tensor, 8, dim=0)[1]
    if case == "split_with_size":
        return torch.split(input_tensor, [8, 8], dim=0)[1]
    if case == "chunk":
        return torch.chunk(input_tensor, 2, dim=0)[1]
    if case == "unbind":
        return torch.unbind(input_tensor, 0)[0]
    raise ValueError(f"unsupported NZ linear view case: {case}")


def _make_quant_matmul_nz_weight_view_inputs(case, shape):
    """Build an NZ weight source whose graph view recovers a quant_matmul weight."""
    m, k, n = shape
    x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8).npu()
    scale = torch.randn(1, dtype=torch.float32).npu()
    weight_base = torch.randint(-5, 5, (k, n), dtype=torch.int8).npu()
    weight_nz = torch_npu.npu_format_cast(weight_base.contiguous(), NZ_FORMAT)

    if case == "permute":
        x2_src = weight_nz.t()
    elif case in ("view", "reshape", "flatten"):
        x2_src = weight_nz.view(2, k // 2, n)
    elif case in ("squeeze", "select", "select_method", "unbind"):
        x2_src = weight_nz.unsqueeze(0)
    else:
        raise ValueError(f"unsupported NZ quant_matmul view case: {case}")

    return x1, x2_src, scale


def _apply_quant_matmul_nz_weight_view(x2_src, case, shape):
    """Recover a 2-D NZ quant_matmul weight through a graph view op."""
    _, k, n = shape
    if case == "permute":
        return x2_src.t()
    if case == "view":
        return x2_src.view(k, n)
    if case == "reshape":
        return x2_src.reshape(k, n)
    if case == "flatten":
        return x2_src.flatten(0, 1)
    if case == "squeeze":
        return x2_src.squeeze(0)
    if case == "select":
        return torch.select(x2_src, 0, 0)
    if case == "select_method":
        return x2_src.select(0, 0)
    if case == "unbind":
        return torch.unbind(x2_src, 0)[0]
    raise ValueError(f"unsupported NZ quant_matmul view case: {case}")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("case,input_shape", LINEAR_VIEW_CASES, ids=[case for case, _ in LINEAR_VIEW_CASES])
def test_nz_view_output_consumed_by_linear_matches_eager(case, input_shape):
    """
    Feature: NZ view outputs consumed by linear
    Description: Verify InferRT view op outputs keep NZ semantics when a following linear accepts the view
    Expectation: Compiled output is numerically consistent with torch_npu eager output
    """

    def func(input_tensor, weight, bias):
        view = _apply_linear_nz_view(input_tensor, case)
        return F.linear(view, weight, bias)

    x_input = _make_linear_nz_view_input(case, input_shape)
    view_input = _apply_linear_nz_view(x_input, case)
    assert torch_npu.get_npu_format(view_input) == NZ_FORMAT

    weight = torch.randn(12, view_input.shape[-1], dtype=torch.float16).npu()
    bias = torch.randn(12, dtype=torch.float16).npu()

    expected_output = func(x_input, weight, bias)
    compiled_func = torch.compile(func, backend=fx_backend, fullgraph=True)
    compiled_output = compiled_func(x_input, weight, bias)

    assert tuple(compiled_output.shape) == tuple(expected_output.shape)
    AssertRtolEqual(expected_output.detach().cpu(), compiled_output.detach().cpu())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("case", QUANT_MATMUL_VIEW_CASES, ids=QUANT_MATMUL_VIEW_CASES)
@pytest.mark.parametrize("shape", QUANT_MATMUL_SHAPES, ids=["shape_8x16x32", "shape_4x32x48", "shape_3x16x64"])
def test_nz_weight_view_output_consumed_by_quant_matmul_matches_eager(case, shape):
    """
    Feature: NZ weight view outputs consumed by quant_matmul
    Description: Verify view ops can recover a valid 2-D NZ weight consumed by quant_matmul
    Expectation: Compiled output is numerically consistent with torch_npu eager output
    """

    def func(x1, x2_src, scale):
        x2 = _apply_quant_matmul_nz_weight_view(x2_src, case, shape)
        return torch_npu.npu_quant_matmul(x1, x2, scale, output_dtype=torch.bfloat16)

    x1, x2_src, scale = _make_quant_matmul_nz_weight_view_inputs(case, shape)
    x2 = _apply_quant_matmul_nz_weight_view(x2_src, case, shape)
    assert x2.is_contiguous()
    assert x2.storage_offset() == 0
    assert torch_npu.get_npu_format(x2) == NZ_FORMAT

    expected_output = func(x1, x2_src, scale)
    compiled_func = torch.compile(func, backend=fx_backend, fullgraph=True)
    compiled_output = compiled_func(x1, x2_src, scale)
    torch.testing.assert_close(compiled_output, expected_output)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_nz_select_method_view_output_consumed_by_linear_matches_eager():
    """
    Feature: NZ Tensor.select method view output consumed by linear
    Description: Verify x.select lowers to select_view before a following NZ-weight linear
    Expectation: Compiled output is numerically consistent with torch_npu eager output
    """

    def func(input_tensor, weight):
        view = input_tensor.select(0, 0)
        return F.linear(view, weight)

    x = torch.randn(1, 4, 16, dtype=torch.bfloat16).npu()
    weight = _make_nz_tensor((6, 16), dtype=torch.bfloat16)
    expected_output = func(x, weight)
    compiled_func = torch.compile(func, backend=fx_backend, fullgraph=True)
    compiled_output = compiled_func(x, weight)

    assert tuple(compiled_output.shape) == tuple(expected_output.shape)
    torch.testing.assert_close(compiled_output, expected_output)
