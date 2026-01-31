"""Tests for torch.sum operation."""
import pytest
import torch

from mrt.torch import fx_mlir_backend as backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def get_sum_func_compiled():
    def sum_func(x, dim=None, keepdim=False, dtype=None):
        return torch.sum(x, dim=dim, keepdim=keepdim, dtype=dtype)

    return torch.compile(sum_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
@pytest.mark.parametrize("shape", ([2,3,4], [12,13,14]))
def test_sum_default(dtype, shape):
    """
    Feature: Test aclnn sum_tensor
    Description: Test aclnn sum_tensor with fp32/fp16 inputs
    Expectation: The result is correct
    """
    sum_func_compiled = get_sum_func_compiled()

    x1 = torch.randn(shape, dtype=dtype).npu()
    output1 = sum_func_compiled(x1, dim=[0], keepdim=True)
    expected1 = torch.sum(x1, dim=[0], keepdim=True)
    AssertRtolEqual(output1, expected1)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_sum_with_dtype(dtype):
    """
    Feature: Test aclnn sum_tensor with dtype
    Description: Test aclnn sum_tensor with fp16/bfloat16 inputs
    Expectation: The result is correct
    """
    sum_func_compiled = get_sum_func_compiled()

    x1 = torch.randn([8, 32], dtype=dtype).npu()
    output1 = sum_func_compiled(x1, dtype=torch.float32)
    expected1 = torch.sum(x1, dtype=torch.float32)
    assert output1.dtype == torch.float32
    AssertRtolEqual(output1, expected1)
