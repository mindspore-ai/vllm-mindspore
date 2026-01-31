"""Tests for tensor setitem operation."""
import pytest
import torch

from mrt.torch.fx_mlir_backend import backend as mlir_backend
from mrt.torch.fx_backend import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (mlir_backend, fx_backend))
def test_tensor_setitem_slice_tensor(backend):
    """
    Feature: Test tensor_setitem_slice_tensor
    Description: Test tensor_setitem_slice_tensor
    Expectation: The result is correct
    """
    def func(x, indices, value):
        res = x.clone()
        res[indices] = value
        return res
    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 2, 3).npu()
    value = torch.randn(2, 2, 3).npu()
    expected = x.clone()
    indices = slice(1, 3, 1)
    out = compiled_op(x, indices, value)
    expected[indices] = value
    AssertRtolEqual(out, expected)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (mlir_backend, fx_backend))
def test_tensor_setitem_tuple_tensor(backend):
    """
    Feature: Test tensor_setitem_tuple_tensor
    Description: Test tensor_setitem_tuple_tensor
    Expectation: The result is correct
    """
    def func(x, value):
        res = x.clone()
        res[0:2, ..., 1:4:2] = value
        return res
    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(3, 2, 4, 4).npu()
    expected = x.clone()
    value = torch.randn(2, 2, 4, 2).npu()
    out = compiled_op(x, value)
    expected[0:2, ..., 1:4:2] = value
    AssertRtolEqual(out, expected)
