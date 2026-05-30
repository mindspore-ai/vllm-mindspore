"""Tests for aclnn bitwise or operation."""
import pytest
import torch

from ms_inferrt.torch import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def bitwise_or_torch(x, y):
    """Using torch.bitwise_or"""
    return torch.bitwise_or(x, y)


def bitwise_or_operator(x, y):
    """Using | operator (covers __or__ and operator.or_)"""
    return x | y


def bitwise_or_method(x, y):
    """Using tensor.bitwise_or method"""
    return x.bitwise_or(y)


def get_op_func_compiled(op_func):
    return torch.compile(op_func, backend=fx_backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", ([2, 3], [15, 64], [1024, 512]))
@pytest.mark.parametrize("dtype", (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool))
@pytest.mark.parametrize("op_func", [bitwise_or_torch, bitwise_or_operator, bitwise_or_method])
def test_bitwise_or(shape, dtype, op_func):
    """
    Feature: Test aclnn bitwise or
    Description: Test aclnn bitwise or with different dtypes, shapes and invocation styles
                 (torch.bitwise_or, | operator, .bitwise_or method)
    Expectation: The result is correct
    """

    if dtype == torch.bool:
        tensor_x = torch.randint(0, 2, shape, dtype=dtype, device="npu")
        tensor_y = torch.randint(0, 2, shape, dtype=dtype, device="npu")
    else:
        # For integer types, use randint for full range coverage
        info = torch.iinfo(dtype)
        tensor_x = torch.randint(info.min, info.max, shape, dtype=dtype, device="npu")
        tensor_y = torch.randint(info.min, info.max, shape, dtype=dtype, device="npu")

    tensor_x_cpu = tensor_x.cpu()
    tensor_y_cpu = tensor_y.cpu()

    result_eager = op_func(tensor_x_cpu, tensor_y_cpu)

    compile_op = get_op_func_compiled(op_func)
    result_compile = compile_op(tensor_x, tensor_y).cpu()

    AssertRtolEqual(result_eager, result_compile)
