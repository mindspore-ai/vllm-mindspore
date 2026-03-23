"""Tests for torch.ops.npu.npu_scatter_nd_update operation."""
import numpy as np
import pytest
import torch

from ms_inferrt.torch.fx_mlir_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(input_tensor, indices_tensor, updates_tensor):
    """Reference implementation of scatter_nd_update."""
    output_tensor = input_tensor.clone()
    for i in range(len(output_tensor)):
        if i < len(indices_tensor):
            output_tensor[indices_tensor[i][0]][indices_tensor[i][1]] = updates_tensor[i]
    return output_tensor


def get_op_func_compiled():
    """Get compiled scatter_nd_update function."""
    def custom_op_func(input_tensor, indices_tensor, updates_tensor):
        return torch.ops.npu.npu_scatter_nd_update(input_tensor, indices_tensor, updates_tensor)
    return torch.compile(custom_op_func, backend=backend)

def run_scatter_nd_update_inner(dtype, op_func_compiled):
    """
    Feature: test scatter_nd_update ops
    Description: precision
    Expectation: success or throw assertion exception.
    Args:
        dtype: The data type of the input.
        op_func_compiled: The compiled op function.
    """
    if dtype == np.float16:
        prec = 0.001
    else:
        prec = 0.0001

    input_tensor_cpu = torch.zeros([3, 2], dtype=dtype)
    indices_tensor_cpu = torch.from_numpy(np.array([[0, 0], [1, 1]]))
    update_tensor_cpu = torch.from_numpy(np.array([10, 20])).to(dtype=dtype)
    input_tensor_npu = input_tensor_cpu.npu()
    indices_tensor_npu = indices_tensor_cpu.npu()
    update_tensor_npu = update_tensor_cpu.npu()

    cpu_output0 = op_func(input_tensor_cpu, indices_tensor_cpu, update_tensor_cpu)
    npu_output = op_func_compiled(input_tensor_npu, indices_tensor_npu, update_tensor_npu)
    npu_output_to_cpu = npu_output.detach().cpu()
    AssertRtolEqual(cpu_output0, npu_output_to_cpu, prec)
    AssertRtolEqual(torch.zeros([3, 2], dtype=dtype), input_tensor_npu.cpu(), prec)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
def test_scatter_nd_update(dtype):
    """
    Feature: Test aclnn scatter_nd_update
    Description: Test aclnn scatter_nd_update with fp32/fp16 inputs
    Expectation: The result is correct
    """
    op_func_compiled = get_op_func_compiled()
    run_scatter_nd_update_inner(dtype, op_func_compiled)
