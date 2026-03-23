"""Tests for aclnn add_rms_norm operation."""
import numpy as np
import torch

from ms_inferrt.torch.fx_mlir_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(x1, x2, gamma, epsilon=1e-6):
    """op function for add_rms_norm"""
    ori_dtype = x1.dtype
    x = x1 + x2
    if ori_dtype == np.float16:
        x = x.astype(np.float32)
    variance = np.mean(np.power(x, 2), axis=-1, keepdims=True)
    std = np.sqrt(variance + epsilon)
    rstd = np.divide(1, std)
    result_mid = x * rstd
    result = result_mid * gamma
    if ori_dtype == np.float16:
        x = x.astype(np.float16)
        result = result.astype(np.float16)
    return result, rstd, x


def get_op_func_compiled():
    def custom_op_func(x1, x2, gamma, epsilon=1e-6):
        return torch.ops.npu.npu_add_rms_norm(x1, x2, gamma, epsilon)
    return torch.compile(custom_op_func, backend=backend)


def add_rms_norm_forward(dtype, op_func_compiled):
    """
    add_rms_norm forward function
    Args:
        dtype: The data type of the input.
        op_func_compiled: The compiled op function.
    """
    if dtype == np.float16:
        prec = 0.001
    else:
        prec = 0.0001

    cpu_input0 = np.random.uniform(0, 100, [1024, 12288]).astype(dtype)
    cpu_input1 = np.random.uniform(0, 100, [1024, 12288]).astype(dtype)
    cpu_input2 = np.random.uniform(0, 100, [12288]).astype(dtype)
    npu_input0 = torch.from_numpy(cpu_input0).npu()
    npu_input1 = torch.from_numpy(cpu_input1).npu()
    npu_input2 = torch.from_numpy(cpu_input2).npu()

    cpu_output0, cpu_output1, cpu_output2 = op_func(cpu_input0, cpu_input1, cpu_input2)
    npu_outputs = op_func_compiled(npu_input0, npu_input1, npu_input2)
    npu_output0, npu_output1, npu_output2 = [
        npu_output.detach().cpu().numpy() for npu_output in npu_outputs
    ]

    AssertRtolEqual(cpu_output0, npu_output0, prec)
    AssertRtolEqual(cpu_output1, npu_output1, prec)
    AssertRtolEqual(cpu_output2, npu_output2, prec)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_add_rms_norm_fp32():
    """
    Feature: Test aclnn add_rms_norm
    Description: Test aclnn add_rms_norm with fp32 inputs
    Expectation: The result is correct
    """
    op_func_compiled = get_op_func_compiled()
    add_rms_norm_forward(np.float32, op_func_compiled)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_add_rms_norm_fp16():
    """
    Feature: Test aclnn add_rms_norm
    Description: Test aclnn add_rms_norm with fp16 inputs
    Expectation: The result is correct
    """
    op_func_compiled = get_op_func_compiled()
    add_rms_norm_forward(np.float16, op_func_compiled)
