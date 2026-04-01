"""Tests for aclnn swiglu operation."""
import pytest
import torch
import torch.nn.functional as F

from ms_inferrt.torch.fx_mlir_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(input_self_tensor, dim=-1):
    x = torch.chunk(input_self_tensor, 2, dim=dim)
    self_tensor = x[0].type(torch.float32)
    other = x[1].type(torch.float32)
    output = F.silu(self_tensor) * other
    return output.type(torch.bfloat16)

def get_op_func_compiled():
    def custom_op_func(input_self_tensor, dim=-1):
        return torch.ops.npu.npu_swiglu(input_self_tensor, dim)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [[8192, 1, 3904*2], [32, 1024]])
def test_swiglu(shape):
    """
    Feature: Test aclnn swiglu
    Description: Test aclnn swiglu with bf16 inputs
    Expectation: The result is correct
    """

    self_tensor = torch.rand(shape, dtype=torch.bfloat16)
    dim = -1
    self_tensor_npu = self_tensor.npu()

    cpu_output = op_func(self_tensor, dim=dim)
    op_func_compiled = get_op_func_compiled()
    npu_output0 = op_func_compiled(self_tensor_npu, dim=dim)
    npu_output = npu_output0.detach().cpu()
    AssertRtolEqual(cpu_output, npu_output)
