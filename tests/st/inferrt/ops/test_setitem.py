"""Tests for tensor setitem operation."""
import pytest
import torch

from ms_inferrt.torch.fx_mlir_backend import backend as mlir_backend
from ms_inferrt.torch.fx_backend import backend as fx_backend

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


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_tensor_setitem_mask_tensor_scalar_value(backend):
    """
    Feature: Test tensor setitem with boolean mask and scalar tensor value
    Description: x[t] = v where t is a boolean mask and v is a scalar tensor input
    Expectation: The result is correct
    """

    def func(x, threshold):
        t = x > threshold
        x[t] = 5
        return x

    compiled_op = torch.compile(func, backend=backend, dynamic=True, fullgraph=False)

    x = torch.ones([2, 3], dtype=torch.int32).npu()
    threshold = torch.tensor(10, dtype=torch.int32).npu()

    expected = func(x, threshold)
    out = compiled_op(x, threshold)
    AssertRtolEqual(out, expected)
