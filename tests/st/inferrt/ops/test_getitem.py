"""Tests for tensor getitem operation."""
import pytest
import torch

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_getitem_tuple_getitem():
    """
    Feature: Test getitem_tuple_getitem
    Description: Test getitem_tuple_getitem
    Expectation: The result is correct
    """
    def func(x, indices):
        return x[indices]
    compiled_op = torch.compile(func, backend=backend)
    x = (2, 3)
    indices = 0
    out = compiled_op(x, indices)
    AssertRtolEqual(out, 2)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_tensor_getitem_by_number():
    """
    Feature: Test tensor_getitem_by_number
    Description: Test tensor_getitem_by_number
    Expectation: The result is correct
    """
    def func(x, indices):
        return x[indices]
    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(2, 3).npu()
    indices = 0
    out = compiled_op(x, indices)
    expected = x[0]
    AssertRtolEqual(out, expected)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_tensor_getitem_by_slice():
    """
    Feature: Test tensor_getitem_by_slice
    Description: Test tensor_getitem_by_slice
    Expectation: The result is correct
    """
    def func(x, indices):
        return x[indices]
    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 3, 2).npu()
    indices = slice(0, 4, 2)
    out = compiled_op(x, indices)
    expected = x[0:4:2]
    AssertRtolEqual(out, expected)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_tensor_getitem_by_tuple():
    """
    Feature: Test tensor_getitem_by_tuple
    Description: Test tensor_getitem_by_tuple
    Expectation: The result is correct
    """
    def func(x):
        return x[1, ..., 1:4:2]
    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 3, 4).npu()
    out = compiled_op(x)
    expected = x[1, ..., 1:4:2]
    AssertRtolEqual(out, expected)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_tensor_getitem_by_tuple_2():
    """
    Feature: Test tensor_getitem_by_tuple
    Description: Test tensor_getitem_by_tuple
    Expectation: The result is correct
    """
    def func(x):
        return x[1, 1:4:2]
    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 4, 6).npu()
    out = compiled_op(x)
    expected = x[1, 1:4:2]
    AssertRtolEqual(out, expected)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_tensor_getitem_by_tuple_with_zero_dim():
    """
    Feature: Test tensor_getitem_by_tuple
    Description: Test tensor_getitem_by_tuple
    Expectation: The result is correct
    """
    def func(x):
        return x[1, :, 4:]
    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 2, 4).npu()
    out = compiled_op(x)
    expected = x[1, :, 4:]
    AssertRtolEqual(out, expected)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_tensor_getitem_by_tensor():
    """
    Feature: Test tensor_getitem_by_tensor
    Description: Test tensor_getitem_by_tensor
    Expectation: The result is correct
    """
    def func(x, indices):
        return x[indices]
    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 3, 2).npu()
    indices = torch.tensor([0, 2]).npu()
    out = compiled_op(x, indices)
    expected = x[indices]
    AssertRtolEqual(out, expected)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_tensor_getitem_by_tuple_with_none(pipeline, monkeypatch):
    """
    Feature: Test tensor_getitem_by_tuple
    Description: Test tensor_getitem_by_tuple
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    def func(x):
        return x[None, 1, None, ..., 1:4:2, None]
    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 4, 6).npu()
    out = compiled_op(x)
    expected = x[None, 1, None, ..., 1:4:2, None]
    AssertRtolEqual(out, expected)
