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
"""Runtime tests for no-op cast elimination in fx_backend."""

import pytest
import torch

from ms_inferrt.torch.cast_elimination import (  # pylint: disable=protected-access
    _is_cast_like_node,
)
from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


_DISABLE_CAST_ELIMINATION_ENV = "MS_INFERRT_DISABLE_CAST_ELIMINATION"


def method_to_dtype(x):
    return x.to(torch.float32)


def method_to_keyword_dtype(x):
    return x.to(dtype=torch.float32)


def method_float(x):
    return x.float()


def method_long(x):
    return x.long()


def prims_convert_element_type(x):
    return torch.ops.prims.convert_element_type.default(x, torch.float32)


def aten_to_dtype(x):
    return torch.ops.aten.to.dtype(x, torch.float32, False, False, None)


def _assert_no_redundant_cast_nodes(gm):
    redundant_casts = [node for node in gm.graph.nodes if _is_cast_like_node(node)]
    assert redundant_casts == []


def _run_and_compare(func, input_tensor, expect_eliminated=True):
    """Compile func and assert its output matches eager, verifying cast elimination status."""
    def backend_with_cast_assertion(gm, example_inputs):
        compiled_callable = backend(gm, example_inputs)
        if expect_eliminated:
            _assert_no_redundant_cast_nodes(gm)
        else:
            assert any(_is_cast_like_node(node) for node in gm.graph.nodes)
        return compiled_callable

    compiled_func = torch.compile(
        func,
        backend=backend_with_cast_assertion,
        fullgraph=True,
    )
    expected = func(input_tensor)
    actual = compiled_func(input_tensor)
    AssertRtolEqual(expected.detach().cpu().numpy(), actual.detach().cpu().numpy())


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize(
    "func,input_dtype",
    [
        (method_to_dtype, torch.float32),
        (method_to_keyword_dtype, torch.float32),
        (method_float, torch.float32),
        (method_long, torch.int64),
        (prims_convert_element_type, torch.float32),
        (aten_to_dtype, torch.float32),
    ],
)
def test_noop_cast_elimination_runtime(func, input_dtype):
    """
    Feature: No-op cast elimination
    Description: Compile no-op dtype casts from supported FX targets
    Expectation: Results match eager and the normalized FX graph has no redundant cast
    """
    if input_dtype.is_floating_point:
        input_tensor = torch.randn((2, 3), dtype=input_dtype, device="npu")
    else:
        input_tensor = torch.randint(-8, 8, (2, 3), dtype=input_dtype, device="npu")

    _run_and_compare(func, input_tensor)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_disable_noop_cast_elimination_runtime(monkeypatch):
    """
    Feature: No-op cast elimination switch
    Description: Compile a redundant cast with the pass disabled
    Expectation: The cast remains in the FX graph and the result matches eager
    """
    monkeypatch.setenv(_DISABLE_CAST_ELIMINATION_ENV, "1")
    input_tensor = torch.randn((2, 3), dtype=torch.float32, device="npu")
    _run_and_compare(method_to_dtype, input_tensor, expect_eliminated=False)
