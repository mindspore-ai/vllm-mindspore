"""Tests for torch.fill_ and torch.copy_ operations."""
import pytest
import torch

from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def get_copy_func_compiled():
    def copy_func(dst, src):
        return dst.copy_(src)
    return torch.compile(copy_func, backend=backend)


def get_fill_func_compiled():
    def fill_func(dst, src):
        return dst.fill_(src)
    return torch.compile(fill_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_copy_scalar(dtype):
    """
    Feature: Test aclnn copy scalar
    Description: Test aclnn copy scalar with fp16/bfloat16 inputs
    Expectation: The result is correct
    """
    dst = torch.randn([2, 3, 4], dtype=dtype).npu()
    src = 2.3
    copy_func_compiled = get_copy_func_compiled()
    copy_func_compiled(dst, src)
    expected_dst = torch.randn([2, 3, 4], dtype=dtype).npu()
    expected_dst.copy_(src)
    AssertRtolEqual(dst, expected_dst)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_fill_scalar(dtype):
    """
    Feature: Test aclnn fill scalar
    Description: Test aclnn fill scalar with fp16/bfloat16 inputs
    Expectation: The result is correct
    """
    dst = torch.randn([4, 6, 32], dtype=dtype).npu()
    src = 2.3
    fill_func_compiled = get_fill_func_compiled()
    fill_func_compiled(dst, src)
    expected_dst = torch.randn([4, 6, 32], dtype=dtype).npu()
    expected_dst.fill_(src)
    AssertRtolEqual(dst, expected_dst)

    dst = torch.randn([2, 6, 16], dtype=dtype).npu()
    fill_func_compiled = get_fill_func_compiled()
    fill_func_compiled(dst, src)
    expected_dst = torch.randn([2, 6, 16], dtype=dtype).npu()
    expected_dst.fill_(src)
    AssertRtolEqual(dst, expected_dst)
