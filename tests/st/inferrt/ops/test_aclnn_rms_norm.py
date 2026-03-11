"""Tests for torch.ops.npu.npu_rms_norm operation."""
import numpy as np
import torch

from ms_inferrt.torch.fx_mlir_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(x, gamma, epsilon=1e-6):
    """Reference implementation of rms_norm."""
    x_fp32 = np.array(x, dtype=np.float32)
    gamma_fp32 = np.array(gamma, dtype=np.float32)

    variance = np.mean(np.power(x_fp32, 2), axis=-1, keepdims=True)
    std = np.sqrt(variance + epsilon)
    rstd = 1 / std
    result_mid = x_fp32 * rstd
    result_fp32 = result_mid * gamma_fp32

    result = np.array(result_fp32, dtype=x.dtype)

    return result, rstd


def get_op_func_compiled():
    """Get compiled rms_norm function."""
    def custom_op_func(x, gamma, epsilon=1e-6):
        return torch.ops.npu.npu_rms_norm(x, gamma, epsilon)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_rms_norm():
    """
    Feature: Test aclnn rms_norm
    Description: Test aclnn rms_norm with fp32 inputs
    Expectation: The result is correct
    """
    cpu_input0 = np.random.uniform(0, 100, [1024, 12288]).astype(np.float32)
    cpu_input1 = np.random.uniform(0, 100, [12288]).astype(np.float32)
    npu_input0 = torch.from_numpy(cpu_input0).npu()
    npu_input1 = torch.from_numpy(cpu_input1).npu()

    cpu_output0, cpu_output1 = op_func(cpu_input0, cpu_input1)
    op_func_compiled = get_op_func_compiled()
    outputs = op_func_compiled(npu_input0, npu_input1)
    npu_output0, npu_output1 = [npu_output.detach().cpu().numpy() for npu_output in outputs]
    AssertRtolEqual(cpu_output0, npu_output0)
    AssertRtolEqual(cpu_output1, npu_output1)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_rms_norm_mix_dtype():
    """
    Feature: Test aclnn rms_norm
    Description: Test aclnn rms_norm with mixed float16 and float32 inputs
    Expectation: The result is correct
    """
    cpu_input0 = np.random.uniform(0, 100, [1024, 12288]).astype(np.float16)
    cpu_input1 = np.random.uniform(0, 100, [12288]).astype(np.float32)
    npu_input0 = torch.from_numpy(cpu_input0).npu()
    npu_input1 = torch.from_numpy(cpu_input1).npu()

    cpu_output0, cpu_output1 = op_func(cpu_input0, cpu_input1)
    op_func_compiled = get_op_func_compiled()
    npu_output_list = [npu_output.detach().cpu().numpy() for npu_output in
                       op_func_compiled(npu_input0, npu_input1)]
    npu_output0, npu_output1 = npu_output_list
    AssertRtolEqual(cpu_output0, npu_output0)
    AssertRtolEqual(cpu_output1, npu_output1)
