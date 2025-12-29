import torch
import pytest
from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual
from mrt.torch.fx_backend import backend

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_getitem_tuple_getitem(pipeline, monkeypatch):
    """
    Feature: Test getitem_tuple_getitem
    Description: Test getitem_tuple_getitem
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    def func(x, indices):
        return x[indices]
    compiled_op = torch.compile(func, backend=backend)
    x = (2, 3)
    indices = 0
    out = compiled_op(x, indices)
    AssertRtolEqual(out, 2)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_tensor_getitem_by_number(pipeline, monkeypatch):
    """
    Feature: Test tensor_getitem_by_number
    Description: Test tensor_getitem_by_number
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    def func(x, indices):
        return x[indices]
    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(2, 3).npu()
    indices = 0
    out = compiled_op(x, indices)
    expected = x[0]
    AssertRtolEqual(out, expected)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_tensor_getitem_by_slice(pipeline, monkeypatch):
    """
    Feature: Test tensor_getitem_by_slice
    Description: Test tensor_getitem_by_slice
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    def func(x, indices):
        return x[indices]
    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 3, 2).npu()
    indices = slice(0, 4, 2)
    out = compiled_op(x, indices)
    expected = x[0:4:2]
    AssertRtolEqual(out, expected)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_tensor_getitem_by_tuple(pipeline, monkeypatch):
    """
    Feature: Test tensor_getitem_by_tuple
    Description: Test tensor_getitem_by_tuple
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    def func(x):
        return x[1, ..., 1:4:2]
    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 3, 4).npu()
    out = compiled_op(x)
    expected = x[1, ..., 1:4:2]
    AssertRtolEqual(out, expected)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_tensor_getitem_by_tuple_2(pipeline, monkeypatch):
    """
    Feature: Test tensor_getitem_by_tuple
    Description: Test tensor_getitem_by_tuple
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    def func(x):
        return x[1, 1:4:2]
    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 4, 6).npu()
    out = compiled_op(x)
    expected = x[1, 1:4:2]
    AssertRtolEqual(out, expected)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_tensor_getitem_by_tuple_with_zero_dim(pipeline, monkeypatch):
    """
    Feature: Test tensor_getitem_by_tuple
    Description: Test tensor_getitem_by_tuple
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    def func(x):
        return x[1, :, 4:]
    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 2, 4).npu()
    out = compiled_op(x)
    expected = x[1, :, 4:]
    AssertRtolEqual(out, expected)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_tensor_getitem_by_tensor(pipeline, monkeypatch):
    """
    Feature: Test tensor_getitem_by_tensor
    Description: Test tensor_getitem_by_tensor
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    def func(x, indices):
        return x[indices]
    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 3, 2).npu()
    indices = torch.tensor([0, 2]).npu()
    out = compiled_op(x, indices)
    expected = x[indices]
    AssertRtolEqual(out, expected)
