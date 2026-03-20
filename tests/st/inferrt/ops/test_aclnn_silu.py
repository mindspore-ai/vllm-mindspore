"""Tests for torch.nn.functional.silu operation."""
import numpy as np
import torch
import torch.nn.functional as F

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(x):
    """Reference implementation of silu."""
    x_fp32 = np.array(x, dtype=np.float32)
    sigmoid_x = 1 / (1 + np.exp(-x_fp32))
    result_fp32 = x_fp32 * sigmoid_x
    result = np.array(result_fp32, dtype=x.dtype)
    return result


def get_op_func_compiled():
    """Get compiled silu function."""
    def custom_op_func(x):
        return F.silu(x)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_silu():
    """
    Feature: Test aclnn silu
    Description: Test aclnn silu with fp32 inputs
    Expectation: The result is correct
    """
    cpu_input0 = np.random.uniform(-10, 10, [1024, 12288]).astype(np.float32)
    npu_input0 = torch.from_numpy(cpu_input0).npu()

    cpu_output0 = op_func(cpu_input0)
    op_func_compiled = get_op_func_compiled()
    npu_output0 = op_func_compiled(npu_input0).detach().cpu().numpy()
    AssertRtolEqual(cpu_output0, npu_output0)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_silu_fp16():
    """
    Feature: Test aclnn silu
    Description: Test aclnn silu with fp16 inputs
    Expectation: The result is correct
    """
    cpu_input0 = np.random.uniform(-10, 10, [1024, 12288]).astype(np.float16)
    npu_input0 = torch.from_numpy(cpu_input0).npu()

    cpu_output0 = op_func(cpu_input0)
    op_func_compiled = get_op_func_compiled()
    npu_output0 = op_func_compiled(npu_input0).detach().cpu().numpy()
    AssertRtolEqual(cpu_output0, npu_output0)
