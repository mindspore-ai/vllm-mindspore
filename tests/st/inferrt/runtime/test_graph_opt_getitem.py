"""Tests for graph optimization with getitem operation."""
import pytest
import torch

from ms_inferrt.torch.fx_backend import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_getitem_const_index(backend):
    """
    Feature: Test getitem with constant index
    Description: Test basic tensor[index] pattern
    Expectation: The result is correct
    """

    def func(x):
        return x[1]

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 3).npu()
    out = compiled_op(x)
    expected = func(x)
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_getitem_tensor_index(backend):
    """
    Feature: Test getitem with tensor index
    Description: Test tensor[tensor_index] pattern
    Expectation: The result is correct
    """

    def func(x, idx):
        return x[idx]

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(6, 3).npu()
    idx = torch.tensor([0, 2, 4], dtype=torch.int64).npu()
    out = compiled_op(x, idx)
    expected = func(x, idx)
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_getitem_tensor_index_composed(backend):
    """
    Feature: Test getitem with computed tensor index
    Description: Index tensor is computed from other tensors
    Expectation: The result is correct
    """

    def func(x, base, offset):
        idx = base + offset
        return x[idx]

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(8, 4).npu()
    base = torch.tensor([0, 2], dtype=torch.int64).npu()
    offset = torch.tensor([1, 3], dtype=torch.int64).npu()
    out = compiled_op(x, base, offset)
    expected = func(x, base, offset)
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_getitem_slice_dynamic(backend):
    """
    Feature: Test getitem with dynamic slice
    Description: Use start/end indices from tensor inputs to slice
    Expectation: The result is correct
    """

    def func(x, start, end):
        return x[start:end, ...]

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(5, 2, 3).npu()
    start = torch.tensor(1, dtype=torch.int64).npu()
    end = torch.tensor(4, dtype=torch.int64).npu()
    out = compiled_op(x, start, end)
    expected = func(x, start, end)
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.skip("Not implemented.")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_getitem_bool_mask(backend):
    """
    Feature: Test getitem with bool mask tensor
    Description: Use boolean mask for indexing
    Expectation: The result is correct
    """

    def func(x, mask):
        return x[mask]

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 3).npu()
    mask = torch.tensor([True, False, True, False]).npu()
    out = compiled_op(x, mask)
    expected = func(x, mask)
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_getitem_with_followup_ops(backend):
    """
    Feature: Test getitem followed by further computation
    Description: getitem result is used in subsequent ops
    Expectation: The result is correct
    """

    def func(x, idx, val2):
        y = x[idx]
        y = y * val2
        return y.sum()

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(6, 3).npu()
    idx = torch.tensor([1, 3, 5], dtype=torch.int64).npu()
    val2 = torch.tensor(2).npu()
    out = compiled_op(x, idx, val2)
    expected = func(x, idx, val2)
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_getitem_multi_output_rmsnorm_first_output(backend):
    """
    Feature: Test getitem on multi-output op (rms_norm)
    Description: Call npu_rms_norm which returns (y, rstd), then take only
                 the first output and use it in further computation.
    Expectation: The result is correct
    """

    def func(x, gamma, val2):
        y, _ = torch.ops.npu.npu_rms_norm(x, gamma, 1e-6)
        return y * val2

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(16, 32).npu()
    gamma = torch.randn(32).npu()
    val2 = torch.tensor(2).npu()
    out = compiled_op(x, gamma, val2)
    expected = func(x, gamma, val2)
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_getitem_multi_output_from_tuple(backend):
    """
    Feature: Test getitem on multi-output op (rms_norm)
    Description: Call npu_rms_norm which returns (y, rstd), then take only
                 the first output and use it in further computation.
    Expectation: The result is correct
    """

    def func(x, gamma, val2):
        x = x + val2
        y = (x, gamma)
        return y[0]

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(16, 32).npu()
    gamma = torch.randn(32).npu()
    val2 = torch.tensor(2).npu()
    out = compiled_op(x, gamma, val2)
    expected = func(x, gamma, val2)
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_getitem_none_and_ellipsis_expand_dims(backend):
    """
    Feature: Test getitem with None and ellipsis for expanding dimensions
    Description: Use x[None, ..., None] style indexing to add dimensions
    Expectation: The result is correct
    """

    def func(x):
        # Add a leading and trailing dimension via None / ellipsis indexing
        y = x[None, ...]        # shape: (1, N, C)
        z = y[..., None]        # shape: (1, N, C, 1)
        return z

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 3).npu()
    out = compiled_op(x)
    expected = func(x)
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_getitem_multiple_none_expand_dims(backend):
    """
    Feature: Test getitem with multiple None and explicit slices
    Description: Use x[None, :, None, :] style indexing to add two new dims
    Expectation: The result is correct
    """

    def func(x):
        return x[None, :, None, :]

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 3).npu()
    out = compiled_op(x)
    expected = func(x)
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_getitem_to_slice_view(backend):
    """
    Feature: Test getitem with None and ellipsis for expanding dimensions
    Description: Use x[None, ..., None] style indexing to add dimensions
    Expectation: The result is correct
    """

    def func(x):
        y = x[..., 4::]
        z = x[..., :4:]
        return torch.sigmoid(z) * y

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(1, 8).npu()
    out = compiled_op(x)
    expected = func(x)
    AssertRtolEqual(out, expected)
    y = torch.randn(1, 8).npu()
    out2 = compiled_op(y)
    expected2 = func(y)
    AssertRtolEqual(out2, expected2)
