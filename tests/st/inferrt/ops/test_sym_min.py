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
"""Tests for torch.sym_min lowering and symbolic Min expressions."""

import pytest
import sympy
import torch

from ms_inferrt.ir import Value
from ms_inferrt.torch.fx_backend import backend as fx_backend
from ms_inferrt.torch.symbolic_shape import SymbolicShapeManager

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def sym_min_func(input_split_sizes):
    limit = input_split_sizes[-1]
    return [torch.sym_min(size, limit) for size in input_split_sizes]


def nested_sym_min_narrow_op(x, y):
    length = torch.sym_min(torch.sym_min(x.shape[0], x.shape[1]), y.shape[0] + y.shape[1])
    return x.narrow(0, 0, length) * 2


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
@pytest.mark.parametrize("input_split_sizes", ([5, 1, 3], [0, 2], [9, 7, 8, 6], [4, 4, 4]))
def test_sym_min_list_comprehension(backend, input_split_sizes):
    """
    Feature: Test torch.sym_min lowering
    Description: Compare eager mode and compiled mode outputs for pure-Python symbolic min computations
    Expectation: The result is correct
    """
    compiled_op = torch.compile(sym_min_func, backend=backend, dynamic=True, fullgraph=False)
    eager_out = sym_min_func(input_split_sizes)
    compiled_out = compiled_op(input_split_sizes)
    AssertRtolEqual(eager_out, compiled_out)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "x_shape,y_shape",
    [
        ((8, 5), (2, 3)),
        ((4, 9), (1, 2)),
        ((6, 7), (10, 1)),
    ],
)
def test_nested_sym_min_shape_narrow_compare_eager(x_shape, y_shape):
    """
    Feature: Test nested torch.sym_min in tensor operator shape
    Description: Compare eager mode and compiled mode outputs when narrow length is min(min(a, b), e + c)
    Expectation: The result is correct
    """
    cpu_x = torch.arange(x_shape[0] * x_shape[1], dtype=torch.float32).reshape(x_shape)
    cpu_y = torch.ones(y_shape, dtype=torch.float32)
    npu_x = cpu_x.npu()
    npu_y = cpu_y.npu()

    compiled_op = torch.compile(nested_sym_min_narrow_op, backend=fx_backend, dynamic=True, fullgraph=False)
    eager_out = nested_sym_min_narrow_op(cpu_x, cpu_y)
    compiled_out = compiled_op(npu_x, npu_y)

    assert eager_out.shape == compiled_out.shape
    AssertRtolEqual(eager_out, compiled_out.cpu())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_sympy_min_to_symbolic_expr():
    """
    Feature: Test sympy.Min conversion to MRT symbolic expression
    Description: Convert a nested sympy Min expression and evaluate it with concrete symbolic variable values
    Expectation: The result is correct
    """
    sym_mgr = SymbolicShapeManager()
    s0, s1 = sympy.symbols("s0 s1", integer=True)

    expr = sym_mgr.convert_sympy_expr_to_symbolic_expr(sympy.Min(s0 + 3, s1 * 2, 10))
    sym_mgr._symbol_map["s0"].set_value(4)  # pylint: disable=protected-access
    sym_mgr._symbol_map["s1"].set_value(3)  # pylint: disable=protected-access

    assert "min(" in repr(expr)
    assert Value(expr).to_int() == min(4 + 3, 3 * 2, 10)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_nested_sympy_min_to_symbolic_expr():
    """
    Feature: Test nested sympy.Min conversion to MRT symbolic expression
    Description: Convert nested min expressions mixed with add/mul expressions and evaluate them
    Expectation: The result is correct
    """
    a, b, c, e = sympy.symbols("a b c e", integer=True)
    cases = [
        (sympy.Min(sympy.Min(a, b, evaluate=False), e + c, evaluate=False), {"a": 8, "b": 5, "c": 2, "e": 4}, 5),
        (sympy.Min(a + 2, sympy.Min(b * 3, c + e, evaluate=False), evaluate=False),
         {"a": 1, "b": 4, "c": 9, "e": 7}, 3),
        (sympy.Min(sympy.Min(a, b, evaluate=False), sympy.Min(c, e, evaluate=False), 12, evaluate=False),
         {"a": 15, "b": 11, "c": 6, "e": 9}, 6),
    ]

    for sympy_expr, values, expected in cases:
        sym_mgr = SymbolicShapeManager()
        expr = sym_mgr.convert_sympy_expr_to_symbolic_expr(sympy_expr)
        for name, value in values.items():
            sym_mgr._symbol_map[name].set_value(value)  # pylint: disable=protected-access

        assert repr(expr).count("min(") >= 2
        assert Value(expr).to_int() == expected
