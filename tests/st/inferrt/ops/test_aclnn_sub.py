import pytest
import numpy as np
import torch

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual
from mrt.torch import fx_mlir_backend as backend


def op_func(x1, x2, alpha):
    """op function for sub"""
    return x1 - alpha * x2


def sub_alpha_2(x1, x2):
    """custom op function with alpha=2"""
    return torch.sub(x1, x2, alpha=2)


def sub_alpha_0_5(x1, x2):
    """custom op function with alpha=0.5"""
    return torch.sub(x1, x2, alpha=0.5)


def sub_no_alpha(x1, x2):
    """custom op function without alpha, using - operator"""
    return x1 - x2


def sub_forward(dtype, shape, alpha, op_func_compiled):
    """
    sub forward function
    Args:
        dtype: The data type of the input.
        alpha: The alpha value in sub.
        op_func_compiled: The compiled op function.
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
    npu_output = op_func_compiled(npu_input0, npu_input1).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output, prec)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
@pytest.mark.parametrize("shape", [(1024, 1024), (256, 512)])
@pytest.mark.parametrize("op_func,alpha", [
    (sub_no_alpha, 1.0),
    (sub_alpha_2, 2),
    (sub_alpha_0_5, 0.5)
])
def test_sub_fp32(pipeline, shape, op_func, alpha, monkeypatch):
    """
    Feature: Test aclnn sub
    Description: Test aclnn sub with fp32 inputs and different alpha types
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    op_func_compiled = torch.compile(op_func, backend=backend)
    sub_forward(np.float32, shape, alpha, op_func_compiled)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
@pytest.mark.parametrize("shape", [(1024, 1024), (256, 512)])
@pytest.mark.parametrize("op_func,alpha", [
    (sub_no_alpha, 1),
    (sub_alpha_2, 2)
])
def test_sub_int32(pipeline, shape, op_func, alpha, monkeypatch):
    """
    Feature: Test aclnn sub
    Description: Test aclnn sub with int32 inputs and different alpha types
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    op_func_compiled = torch.compile(op_func, backend=backend)
    sub_forward(np.int32, shape, alpha, op_func_compiled)
