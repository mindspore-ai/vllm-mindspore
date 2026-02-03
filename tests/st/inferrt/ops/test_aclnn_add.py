"""Tests for aclnn add operation."""
import numpy as np
import pytest
import torch

from mrt.torch import fx_mlir_backend as backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(x1, x2, alpha):
    """op function for add"""
    return x1 + alpha * x2


def add_alpha_2(x1, x2):
    """custom op function with alpha=2"""
    return torch.add(x1, x2, alpha=2)


def add_alpha_0_5(x1, x2):
    """custom op function with alpha=0.5"""
    return torch.add(x1, x2, alpha=0.5)


def add_no_alpha(x1, x2):
    """custom op function without alpha, using + operator"""
    return x1 + x2


def add_forward(dtype, shape, alpha, compiled_func):
    """
    add forward function
    Args:
        dtype: The data type of the input.
        alpha: The alpha value in add.
        compiled_func: The compiled op function.
    """
    if np.issubdtype(dtype, np.integer):
        cpu_input0 = np.random.randint(-100, 100, shape).astype(dtype)
        cpu_input1 = np.random.randint(-100, 100, shape).astype(dtype)
        prec = 0
    else:
        if dtype == np.float16:
            prec = 0.001
        else:
            prec = 0.0001
        cpu_input0 = np.random.uniform(-1, 1, shape).astype(dtype)
        cpu_input1 = np.random.uniform(-1, 1, shape).astype(dtype)

    npu_input0 = torch.from_numpy(cpu_input0).npu()
    npu_input1 = torch.from_numpy(cpu_input1).npu()

    cpu_output = op_func(cpu_input0, cpu_input1, alpha)
    npu_output = compiled_func(npu_input0, npu_input1).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output, prec)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(1024, 1024), (256, 512)])
@pytest.mark.parametrize("op_func,alpha", [
    (add_no_alpha, 1.0),
    (add_alpha_2, 2),
    (add_alpha_0_5, 0.5)
])
# pylint: disable=redefined-outer-name
def test_add_fp32(shape, op_func, alpha):
    """
    Feature: Test aclnn add
    Description: Test aclnn add with fp32 inputs and different alpha types
    Expectation: The result is correct
    """
    compiled_op = torch.compile(op_func, backend=backend)
    add_forward(np.float32, shape, alpha, compiled_op)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(1024, 1024), (256, 512)])
@pytest.mark.parametrize("op_func,alpha", [
    (add_no_alpha, 1),
    (add_alpha_2, 2)
])
# pylint: disable=redefined-outer-name
def test_add_int32(shape, op_func, alpha):
    """
    Feature: Test aclnn add
    Description: Test aclnn add with int32 inputs and different alpha types
    Expectation: The result is correct
    """
    compiled_op = torch.compile(op_func, backend=backend)
    add_forward(np.int32, shape, alpha, compiled_op)
