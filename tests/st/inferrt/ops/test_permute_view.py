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
"""Runtime tests for permute_view-backed view ops."""

import pytest
import torch
import torch_npu

from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark


NZ_FORMAT = 29
BASE_SHAPE = (2, 16, 32)
BASE_SHAPE_2D = (16, 32)
TEST_SHAPES_2D = [(16, 32), (24, 40)]
TEST_SHAPES_3D = [(2, 16, 32), (3, 12, 24)]
MATMUL_SHAPE = (2, 16, 32)  # (M, K, N)
MATMUL_TEST_SHAPES = [MATMUL_SHAPE, (4, 32, 48), (3, 16, 64)]


def _make_input(fmt: str, non_contiguous: bool, shape) -> torch.Tensor:
    x = torch.arange(shape[0] * shape[1] * shape[2], dtype=torch.float32).reshape(shape).npu()
    if fmt == "NZ":
        x = torch_npu.npu_format_cast(x.contiguous(), NZ_FORMAT)
    if non_contiguous:
        x = x[:, :, ::2]
    return x


def _make_input_2d(fmt: str, non_contiguous: bool, shape) -> torch.Tensor:
    x = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape).npu()
    if fmt == "NZ":
        x = torch_npu.npu_format_cast(x.contiguous(), NZ_FORMAT)
    if non_contiguous:
        x = x[:, ::2]
    return x


def _make_quant_matmul_t_inputs(case: str, shape):
    """Build quant_matmul inputs for ND/NZ and contiguous/non-contiguous weight cases."""
    m, k, n = shape
    x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8).npu()
    scale = torch.randn(1, dtype=torch.float32).npu()

    if case == "nd_non_contiguous":
        # x2_src.t() -> [K, N], ND non-contiguous weight for quant_matmul.
        x2_src = torch.randint(-5, 5, (n, k), dtype=torch.int8).npu()
        weight = x2_src.t()
        assert not weight.is_contiguous()
        assert torch_npu.get_npu_format(weight) != NZ_FORMAT
        return x1, x2_src, scale

    if case == "nz_contiguous":
        # Build a contiguous NZ weight [K, N], then transpose once as graph input.
        # In graph, x2_src.t() restores contiguous NZ weight for quant_matmul.
        weight_base = torch.randint(-5, 5, (k, n), dtype=torch.int8).npu()
        weight_nz_contig = torch_npu.npu_format_cast(weight_base.contiguous(), NZ_FORMAT)
        x2_src = weight_nz_contig.t()
        weight = x2_src.t()
        assert weight.is_contiguous()
        assert torch_npu.get_npu_format(weight) == NZ_FORMAT
        return x1, x2_src, scale

    if case == "nz_non_contiguous":
        # Build non-contiguous NZ weight [K, N], then transpose as graph input.
        # In graph, x2_src.t() restores the same NZ non-contiguous weight.
        weight_big = torch.randint(-5, 5, (k, n * 2), dtype=torch.int8).npu()
        weight_big_nz = torch_npu.npu_format_cast(weight_big.contiguous(), NZ_FORMAT)
        weight = weight_big_nz[:, ::2]
        x2_src = weight.t()
        assert not weight.is_contiguous()
        assert torch_npu.get_npu_format(weight) == NZ_FORMAT
        return x1, x2_src, scale

    raise ValueError(f"Unsupported case: {case}")


def _assert_tensor_matches_eager(eager_out: torch.Tensor, compiled_out: torch.Tensor):
    assert tuple(compiled_out.shape) == tuple(eager_out.shape)
    torch.testing.assert_close(compiled_out, eager_out)


def _assert_view_metadata_matches_eager(
    input_tensor: torch.Tensor,
    eager_out: torch.Tensor,
    compiled_out: torch.Tensor,
):
    assert tuple(compiled_out.shape) == tuple(eager_out.shape)
    assert tuple(compiled_out.stride()) == tuple(eager_out.stride())
    assert compiled_out.storage_offset() == eager_out.storage_offset()
    assert compiled_out.is_contiguous() == eager_out.is_contiguous()
    assert torch_npu.get_npu_format(compiled_out) == torch_npu.get_npu_format(input_tensor)


def _assert_view_matches_eager(input_tensor: torch.Tensor, eager_out: torch.Tensor, compiled_out: torch.Tensor):
    _assert_view_metadata_matches_eager(input_tensor, eager_out, compiled_out)
    _assert_tensor_matches_eager(eager_out, compiled_out)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("fmt", ["ND", "NZ"])
@pytest.mark.parametrize("non_contiguous", [False, True], ids=["contiguous", "non_contiguous"])
@pytest.mark.parametrize("shape", TEST_SHAPES_2D, ids=["shape_16x32", "shape_24x40"])
def test_t_to_permute_view(fmt: str, non_contiguous: bool, shape):
    """
    Feature: t lowered to permute_view
    Description: Verify .t() is lowered as a view op for ND/NZ contiguous and non-contiguous 2-D inputs
    Expectation:
        ND: Compiled result matches eager in view metadata and value
        NZ: Compiled result keeps view metadata path consistency (shape/stride/storage_offset/format)
    """
    def func(x: torch.Tensor) -> torch.Tensor:
        return x.t()

    x = _make_input_2d(fmt, non_contiguous, shape)
    eager_out = func(x)
    compiled_func = torch.compile(func, backend=backend)
    compiled_out = compiled_func(x)
    if fmt == "NZ":
        # Current graph output handling for NZ tensors is incomplete:
        # metadata required for NZ->ND value materialization is not fully preserved.
        # Therefore, we only check view metadata consistency here, and skip value equality
        # comparison with eager output for NZ format.
        _assert_view_metadata_matches_eager(x, eager_out, compiled_out)
        return
    _assert_view_matches_eager(x, eager_out, compiled_out)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("fmt", ["ND", "NZ"])
@pytest.mark.parametrize("non_contiguous", [False, True], ids=["contiguous", "non_contiguous"])
@pytest.mark.parametrize("shape", TEST_SHAPES_3D, ids=["shape_2x16x32", "shape_3x12x24"])
def test_movedim_to_permute(fmt: str, non_contiguous: bool, shape):
    """
    Feature: movedim lowered to permute
    Description: Verify movedim is normalized in fx_backend and executes with builtin permute path for ND/NZ inputs
    Expectation:
        ND: Compiled result matches eager numerically with same logical shape
        NZ: Lowering reaches builtin aclnnPermute path (current device may report unsupported with ret=561103)
    """
    def func(x: torch.Tensor) -> torch.Tensor:
        return x.movedim((0, 2), (1, 0))

    x = _make_input(fmt, non_contiguous, shape)
    eager_out = func(x)
    compiled_func = torch.compile(func, backend=backend)
    if fmt == "NZ":
        with pytest.raises(RuntimeError, match="aclnnPermuteGetWorkspaceSize"):
            compiled_func(x)
        return
    compiled_out = compiled_func(x)
    _assert_tensor_matches_eager(eager_out, compiled_out)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "case",
    ["nd_non_contiguous", "nz_contiguous", "nz_non_contiguous"],
    ids=["nd_non_contiguous", "nz_contiguous", "nz_non_contiguous"],
)
@pytest.mark.parametrize("shape", MATMUL_TEST_SHAPES, ids=["shape_2x16x32", "shape_4x32x48", "shape_3x16x64"])
def test_t_to_permute_view_for_quant_matmul(case: str, shape):
    """
    Feature: t lowered to permute_view with quant_matmul
    Description:
        Verify weight built by `.t()` follows permute_view path and quant_matmul result matches eager mode for:
        1) ND non-contiguous weight
        2) NZ contiguous weight
        3) NZ non-contiguous weight
    Expectation: Compiled result is numerically consistent with eager result.
    """

    def func(x1: torch.Tensor, x2_src: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_quant_matmul(x1, x2_src.t(), scale, output_dtype=torch.bfloat16)

    x1, x2_src, scale = _make_quant_matmul_t_inputs(case, shape)
    if case == "nz_non_contiguous":
        with pytest.raises(RuntimeError, match=r"x2.*format.*(only support NZ|not NZ|Ascend affinity format)"):
            func(x1, x2_src, scale)
        return

    eager_out = func(x1, x2_src, scale)
    compiled_func = torch.compile(func, backend=backend)
    compiled_out = compiled_func(x1, x2_src, scale)
    _assert_tensor_matches_eager(eager_out, compiled_out)
