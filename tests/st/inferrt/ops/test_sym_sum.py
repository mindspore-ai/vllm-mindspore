"""Tests for torch.sym_sum lowering in FX backend."""

import pytest
import torch

from mrt.torch.fx_backend import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def func(input_split_sizes):
    send_displacements = [sum(input_split_sizes[:i]) for i in range(len(input_split_sizes))]
    return send_displacements


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
@pytest.mark.parametrize("input_split_sizes", ([1, 2, 3], [0, 0, 5, 1], [7], [3, 0, 0, 2]))
def test_sym_sum_list_comprehension(backend, input_split_sizes):
    """
    Feature: Test torch.sym_sum lowering
    Description: Compare eager mode and compiled mode outputs for pure-Python list shape computations
    Expectation: The result is correct
    """
    compiled_op = torch.compile(func, backend=backend, dynamic=True, fullgraph=False)
    eager_out = func(input_split_sizes)
    compiled_out = compiled_op(input_split_sizes)
    AssertRtolEqual(eager_out, compiled_out)
