"""Tests for torch.masked_fill operation."""
import pytest
import torch

from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def get_masked_fill_func_compiled():
    def masked_fill_func(x, mask, value):
        return torch.masked_fill(x, mask, value)
    return torch.compile(masked_fill_func, backend=backend)


def get_inplace_masked_fill_func_compiled():
    def inplace_masked_fill_func(x, mask, value):
        return x.masked_fill_(mask, value)
    return torch.compile(inplace_masked_fill_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
def test_masked_fill_tensor(dtype):
    """
    Feature: Test aclnn masked_fill_tensor
    Description: Test aclnn masked_fill_tensor with fp32/fp16 inputs
    Expectation: The result is correct
    """
    masked_fill_func_compiled = get_masked_fill_func_compiled()

    x1 = torch.randn([3, 2], dtype=dtype).npu()
    mask1 = torch.tensor([[True, False], [False, True], [True, True]]).npu()
    value1 = torch.tensor(10.0, dtype=dtype).npu()
    output1 = masked_fill_func_compiled(x1, mask1, value1)
    expected1 = torch.masked_fill(x1, mask1, value1)
    AssertRtolEqual(output1, expected1)

    x2 = torch.randn([16, 32], dtype=dtype).npu()
    mask2 = torch.randint(0, 2, [16, 32], dtype=torch.bool).npu()
    value2 = torch.tensor(10.0, dtype=dtype).npu()
    output2 = masked_fill_func_compiled(x2, mask2, value2)
    expected2 = torch.masked_fill(x2, mask2, value2)
    AssertRtolEqual(output2, expected2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
def test_inplace_masked_fill_tensor(dtype):
    """
    Feature: Test aclnn inplace_masked_fill_tensor
    Description: Test aclnn inplace_masked_fill_tensor with fp32/fp16 inputs
    Expectation: The result is correct
    """
    inplace_masked_fill_func_compiled = get_inplace_masked_fill_func_compiled()

    x1 = torch.randn([3, 2], dtype=dtype).npu()
    expected_x1 = x1.clone()
    mask1 = torch.tensor([[True, False], [False, True], [True, True]]).npu()
    value1 = torch.tensor(10.0, dtype=dtype).npu()

    inplace_masked_fill_func_compiled(x1, mask1, value1)
    expected_x1.masked_fill_(mask1, value1)
    AssertRtolEqual(x1, expected_x1)

    x2 = torch.randn([16, 32], dtype=dtype).npu()
    expected_x2 = x2.clone()
    mask2 = torch.randint(0, 2, [16, 32], dtype=torch.bool).npu()
    value2 = torch.tensor(10.0, dtype=dtype).npu()
    inplace_masked_fill_func_compiled(x2, mask2, value2)
    expected_x2.masked_fill_(mask2, value2)
    AssertRtolEqual(x2, expected_x2)
