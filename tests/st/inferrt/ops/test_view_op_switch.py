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
"""Tests for disabling selected InferRT view op lowering."""

import torch
import torch._dynamo.config as dynamo_config

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark  # pylint: disable=import-error


dynamo_config.cache_size_limit = 64
_DISABLE_VIEW_OPS_ENV = "MS_INFERRT_DISABLE_VIEW_OPS"


def _assert_compiled_matches_eager(func, *args):
    torch.compiler.reset()
    try:
        compiled_func = torch.compile(func, backend=backend, fullgraph=True)
        expected = func(*args)
        actual = compiled_func(*args)
        torch.testing.assert_close(actual, expected)
    finally:
        torch.compiler.reset()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_disable_permute_view_switch(monkeypatch):
    """
    Feature: InferRT view op switch
    Description: Disable permute_view and run torch.permute through non-view fallback
    Expectation: Compiled output matches eager mode
    """
    monkeypatch.setenv(_DISABLE_VIEW_OPS_ENV, "permute")

    def func(x):
        return torch.permute(x, (2, 0, 1)) + 1.0

    x = torch.randn((2, 3, 4), dtype=torch.float32).npu()
    _assert_compiled_matches_eager(func, x)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_disable_transpose_view_switch(monkeypatch):
    """
    Feature: InferRT view op switch
    Description: Disable permute_view selected by torch.transpose and run through non-view fallback
    Expectation: Compiled output matches eager mode
    """
    monkeypatch.setenv(_DISABLE_VIEW_OPS_ENV, "transpose")

    def func(x):
        return torch.transpose(x, 0, 1) + 1.0

    x = torch.randn((2, 3, 4), dtype=torch.float32).npu()
    _assert_compiled_matches_eager(func, x)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_disable_split_tensor_view_switch(monkeypatch):
    """
    Feature: InferRT view op switch
    Description: Disable split view lowering for integer split size
    Expectation: Compiled output matches eager mode
    """
    monkeypatch.setenv(_DISABLE_VIEW_OPS_ENV, "split")

    def func(x):
        first, second, third = torch.split(x, 2, dim=0)
        return first + second + third

    x = torch.randn((6, 3), dtype=torch.float32).npu()
    _assert_compiled_matches_eager(func, x)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_disable_view_without_safe_fallback_keeps_view(monkeypatch):
    """
    Feature: InferRT view op switch
    Description: Disable squeeze_view, which has no supported non-view fallback in the view switch
    Expectation: Compiled output matches eager mode by keeping the view implementation
    """
    monkeypatch.setenv(_DISABLE_VIEW_OPS_ENV, "squeeze")

    def func(x):
        return torch.squeeze(x, 0) + 1.0

    x = torch.randn((1, 3, 4), dtype=torch.float32).npu()
    _assert_compiled_matches_eager(func, x)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_disable_split_with_size_view_switch(monkeypatch):
    """
    Feature: InferRT view op switch
    Description: Disable split view lowering for explicit split sizes
    Expectation: Compiled output matches eager mode
    """
    monkeypatch.setenv(_DISABLE_VIEW_OPS_ENV, "split_with_size")

    def func(x):
        first, second = torch.split(x, [2, 4], dim=0)
        return torch.cat((first, second), dim=0)

    x = torch.randn((6, 3), dtype=torch.float32).npu()
    _assert_compiled_matches_eager(func, x)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_disable_chunk_view_switch(monkeypatch):
    """
    Feature: InferRT view op switch
    Description: Disable chunk lowering to split_with_size_view while keeping chunk argument normalization
    Expectation: Compiled output matches eager mode
    """
    monkeypatch.setenv(_DISABLE_VIEW_OPS_ENV, "chunk")

    def func(x):
        first, second, third = torch.chunk(x, 3, dim=0)
        return first + second + third

    x = torch.randn((6, 3), dtype=torch.float32).npu()
    _assert_compiled_matches_eager(func, x)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_disable_tensor_chunk_view_switch(monkeypatch):
    """
    Feature: InferRT view op switch
    Description: Disable Tensor.chunk lowering to split_with_size_view and fall back to split_with_size
    Expectation: Compiled output matches eager mode
    """
    monkeypatch.setenv(_DISABLE_VIEW_OPS_ENV, "chunk")

    def func(x):
        first, second, third = x.chunk(3, dim=0)
        return first + second + third

    x = torch.randn((6, 3), dtype=torch.float32).npu()
    _assert_compiled_matches_eager(func, x)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_disable_multiple_view_ops_switch(monkeypatch):
    """
    Feature: InferRT view op switch
    Description: Disable multiple view ops in one environment variable
    Expectation: Compiled output matches eager mode
    """
    monkeypatch.setenv(_DISABLE_VIEW_OPS_ENV, "permute,split,chunk")

    def func(x, y):
        first, second, third = torch.split(x, 2, dim=0)
        moved = torch.permute(y, (2, 0, 1))
        return first + second + third, moved + 1.0

    x = torch.randn((6, 3), dtype=torch.float32).npu()
    y = torch.randn((2, 3, 4), dtype=torch.float32).npu()
    _assert_compiled_matches_eager(func, x, y)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_disable_multiple_permute_api_view_switch(monkeypatch):
    """
    Feature: InferRT view op switch
    Description: Disable transpose and movedim view lowering with API-level switch names
    Expectation: Compiled output matches eager mode
    """
    monkeypatch.setenv(_DISABLE_VIEW_OPS_ENV, "transpose,movedim")

    def func(x, y):
        return torch.transpose(x, 0, 1) + 1.0, torch.movedim(y, 0, 2) + 1.0

    x = torch.randn((2, 3, 4), dtype=torch.float32).npu()
    y = torch.randn((2, 3, 4), dtype=torch.float32).npu()
    _assert_compiled_matches_eager(func, x, y)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_disable_tensor_t_view_switch(monkeypatch):
    """
    Feature: InferRT view op switch
    Description: Disable permute_view selected by Tensor.t() and run through non-view fallback
    Expectation: Compiled output matches eager mode
    """
    monkeypatch.setenv(_DISABLE_VIEW_OPS_ENV, "t")

    def func(x):
        return x.t() + 1.0

    x = torch.randn((3, 4), dtype=torch.float32).npu()
    _assert_compiled_matches_eager(func, x)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_disable_aten_permute_view_switch(monkeypatch):
    """
    Feature: InferRT view op switch
    Description: Disable permute_view selected by aten.permute.default and run through non-view fallback
    Expectation: Compiled output matches eager mode
    """
    monkeypatch.setenv(_DISABLE_VIEW_OPS_ENV, "permute")

    def func(x):
        return torch.ops.aten.permute.default(x, [2, 0, 1]) + 1.0

    x = torch.randn((2, 3, 4), dtype=torch.float32).npu()
    _assert_compiled_matches_eager(func, x)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_disable_aten_transpose_view_switch(monkeypatch):
    """
    Feature: InferRT view op switch
    Description: Disable permute_view selected by aten.transpose.int and run through non-view fallback
    Expectation: Compiled output matches eager mode
    """
    monkeypatch.setenv(_DISABLE_VIEW_OPS_ENV, "transpose")

    def func(x):
        return torch.ops.aten.transpose.int(x, 0, 1) + 1.0

    x = torch.randn((2, 3, 4), dtype=torch.float32).npu()
    _assert_compiled_matches_eager(func, x)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_disable_aten_t_view_switch(monkeypatch):
    """
    Feature: InferRT view op switch
    Description: Disable permute_view selected by aten.t.default and run through non-view fallback
    Expectation: Compiled output matches eager mode
    """
    monkeypatch.setenv(_DISABLE_VIEW_OPS_ENV, "t")

    def func(x):
        return torch.ops.aten.t.default(x) + 1.0

    x = torch.randn((3, 4), dtype=torch.float32).npu()
    _assert_compiled_matches_eager(func, x)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_disable_aten_movedim_view_switch(monkeypatch):
    """
    Feature: InferRT view op switch
    Description: Disable permute_view selected by aten.movedim.int and run through non-view fallback
    Expectation: Compiled output matches eager mode
    """
    monkeypatch.setenv(_DISABLE_VIEW_OPS_ENV, "movedim")

    def func(x):
        return torch.ops.aten.movedim.int(x, 0, 2) + 1.0

    x = torch.randn((2, 3, 4), dtype=torch.float32).npu()
    _assert_compiled_matches_eager(func, x)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_disable_aten_flatten_without_safe_fallback_keeps_view(monkeypatch):
    """
    Feature: InferRT view op switch
    Description: Disable flatten_view selected by aten.flatten.using_ints without a non-view fallback
    Expectation: Compiled output matches eager mode by keeping the view implementation
    """
    monkeypatch.setenv(_DISABLE_VIEW_OPS_ENV, "flatten")

    def func(x):
        return torch.ops.aten.flatten.using_ints(x, 0, 1) + 1.0

    x = torch.randn((2, 3, 4), dtype=torch.float32).npu()
    _assert_compiled_matches_eager(func, x)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_disable_aten_chunk_view_switch(monkeypatch):
    """
    Feature: InferRT view op switch
    Description: Disable aten.chunk.default lowering to split_with_size_view and fall back to split_with_size
    Expectation: Compiled output matches eager mode
    """
    monkeypatch.setenv(_DISABLE_VIEW_OPS_ENV, "chunk")

    def func(x):
        first, second, third = torch.ops.aten.chunk.default(x, 3, 0)
        return first + second + third

    x = torch.randn((6, 3), dtype=torch.float32).npu()
    _assert_compiled_matches_eager(func, x)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_disable_aten_split_default_view_switch(monkeypatch):
    """
    Feature: InferRT view op switch
    Description: Disable split_with_size_view selected by aten.split.default and fall back to split_with_size
    Expectation: Compiled output matches eager mode
    """
    monkeypatch.setenv(_DISABLE_VIEW_OPS_ENV, "split")

    def func(x):
        first, second = torch.ops.aten.split.default(x, [2, 4], 0)
        return torch.cat((first, second), dim=0)

    x = torch.randn((6, 3), dtype=torch.float32).npu()
    _assert_compiled_matches_eager(func, x)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_disable_aten_split_sizes_view_switch(monkeypatch):
    """
    Feature: InferRT view op switch
    Description: Disable split_with_size_view selected by aten.split.sizes and fall back to split_with_size
    Expectation: Compiled output matches eager mode
    """
    monkeypatch.setenv(_DISABLE_VIEW_OPS_ENV, "split")

    def func(x):
        first, second = torch.ops.aten.split.sizes(x, [2, 4], 0)
        return torch.cat((first, second), dim=0)

    x = torch.randn((6, 3), dtype=torch.float32).npu()
    _assert_compiled_matches_eager(func, x)
