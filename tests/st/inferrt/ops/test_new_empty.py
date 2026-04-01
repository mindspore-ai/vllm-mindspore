"""Tests for torch.tensor.new_empty operation."""
import pytest
import torch

from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(self, size, dtype):
    return self.new_empty(size, dtype=dtype)


def get_op_func_compiled():
    def custom_op_func(self, size, dtype, device=torch.device('npu')):
        return self.new_empty(size, dtype=dtype, device=device)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("self_shape", [[10, 10], [20, 30, 35]])
@pytest.mark.parametrize("new_shape", [[5, 5], [10, 20, 25], [100]])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_new_empty_basic(self_shape, new_shape, dtype):
    """
    Feature: Test torch.tensor.new_empty
    Description: Test new_empty with different shapes and dtypes
    Expectation: The result shape and dtype are correct
    """

    self_tensor = torch.randn(self_shape, dtype=dtype).npu()
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(self_tensor, new_shape, dtype)
    AssertRtolEqual(torch.Size(new_shape), npu_output.shape)
    assert npu_output.dtype == dtype, f"dtype should be {dtype}, but got {npu_output.dtype}"
    assert npu_output.device.type == 'npu', f"device should be npu, but got {npu_output.device.type}"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("self_shape", [[64, 128], [256, 512]])
@pytest.mark.parametrize("new_shape", [[32, 64], [128, 256, 128]])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_new_empty_int_dtype(self_shape, new_shape, dtype):
    """
    Feature: Test torch.tensor.new_empty with integer dtypes
    Description: Test new_empty with int32 and int64 dtypes
    Expectation: The result shape and dtype are correct
    """

    self_tensor = torch.randn(self_shape, dtype=torch.float32).npu()
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(self_tensor, new_shape, dtype)
    AssertRtolEqual(torch.Size(new_shape), npu_output.shape)
    assert npu_output.dtype == dtype, f"dtype should be {dtype}, but got {npu_output.dtype}"
    assert npu_output.device.type == 'npu', f"device should be npu, but got {npu_output.device.type}"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("self_shape", [[10], [20, 30]])
@pytest.mark.parametrize("new_shape", [[], [1], [5, 6, 7]])
def test_new_empty_same_dtype(self_shape, new_shape):
    """
    Feature: Test torch.tensor.new_empty with same dtype as self
    Description: Test new_empty inheriting dtype from self tensor
    Expectation: The result has the same dtype as self tensor
    """

    dtype = torch.float32
    self_tensor = torch.randn(self_shape, dtype=dtype).npu()
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(self_tensor, new_shape, dtype)
    AssertRtolEqual(torch.Size(new_shape), npu_output.shape)
    assert npu_output.dtype == dtype, f"dtype should be {dtype}, but got {npu_output.dtype}"
    assert npu_output.device.type == 'npu', f"device should be npu, but got {npu_output.device.type}"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("self_shape", [[10, 10], [128, 256]])
@pytest.mark.parametrize("new_shape", [[20, 20], [64, 128, 64]])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_new_empty_different_self_dtype(self_shape, new_shape, dtype):
    """
    Feature: Test torch.tensor.new_empty with different self tensor dtype
    Description: Test new_empty when self tensor has different dtype than output
    Expectation: The output dtype matches the specified dtype, not self's dtype
    """

    self_dtype = torch.bfloat16 if dtype == torch.float16 else torch.float16
    self_tensor = torch.randn(self_shape, dtype=self_dtype)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(self_tensor, new_shape, dtype, device=torch.device('cpu'))
    AssertRtolEqual(torch.Size(new_shape), npu_output.shape)
    assert npu_output.dtype == dtype, f"dtype should be {dtype}, but got {npu_output.dtype}"
    assert npu_output.device.type == 'cpu', f"device should be npu, but got {npu_output.device.type}"
