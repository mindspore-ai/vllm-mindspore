"""Tests for torch.copy_ operation."""
import pytest
import torch

from mrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def get_copy_func_compiled():
    def copy_func(dst, src, non_blocking=False):
        return dst.copy_(src, non_blocking=non_blocking)
    return torch.compile(copy_func, backend=backend)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_copy_tensor_dynamic_shape(dtype):
    """
    Feature: Test aclnn copy_tensor
    Description: Test aclnn copy_tensor with fp32/fp16/bfloat16 inputs
    Expectation: The result is correct
    """
    dst = torch.randn([2, 3, 4], dtype=dtype).npu()
    src = torch.randn([2, 3, 4], dtype=dtype).npu()
    copy_func_compiled = get_copy_func_compiled()
    copy_func_compiled(dst, src)
    expected1_dst = torch.randn([2, 3, 4], dtype=dtype).npu()
    expected1_dst.copy_(src)
    AssertRtolEqual(dst, expected1_dst)

    dst2 = torch.randn([12, 13, 14], dtype=dtype).npu()
    src2 = torch.randn([12, 13, 14], dtype=dtype).npu()
    copy_func_compiled(dst2, src2)
    expected2_dst = torch.randn([12, 13, 14], dtype=dtype).npu()
    expected2_dst.copy_(src2)
    AssertRtolEqual(dst2, expected2_dst)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_copy_host_to_device(dtype):
    """
    Feature: Test aclnn copy tensor from host to device
    Description: Test aclnn copy tensor from host to device with fp16/bfloat16 inputs
    Expectation: The result is correct
    """
    dst = torch.randn([2, 3, 4], dtype=dtype).npu()
    src = torch.randn([2, 3, 4], dtype=dtype).cpu()
    copy_func_compiled = get_copy_func_compiled()
    copy_func_compiled(dst, src)
    expected_dst = torch.randn([2, 3, 4], dtype=dtype).npu()
    expected_dst.copy_(src)
    AssertRtolEqual(dst, expected_dst)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", (torch.float16, torch.int32))
def test_copy_tensor_with_non_blocking(dtype):
    """
    Feature: Test aclnn copy_tensor
    Description: Test aclnn copy_tensor with fp16/int32 inputs
    Expectation: The result is correct
    """
    dst = torch.randint(-100, 100, [32, 64], dtype=dtype).npu()
    src = torch.randint(-100, 100, [32, 64], dtype=dtype).cpu()
    copy_func_compiled = get_copy_func_compiled()
    copy_func_compiled(dst, src, non_blocking=True)
    expected1_dst = torch.randint(-100, 100, [32, 64], dtype=dtype).npu()
    expected1_dst.copy_(src, non_blocking=True)
    AssertRtolEqual(dst, expected1_dst)
