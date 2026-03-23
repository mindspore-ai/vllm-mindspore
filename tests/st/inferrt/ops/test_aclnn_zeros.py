"""Tests for torch.zeros operation."""
import pytest
import torch

from ms_inferrt.torch import fx_mlir_backend as backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual

def zeros_func(size, dtype):
    return torch.zeros(size, dtype=dtype)

def get_zeros_op_func_compiled():
    def custom_op_func(size, dtype, device=torch.device('npu')):
        return torch.zeros(size, dtype=dtype, device=device)
    return torch.compile(custom_op_func, backend=backend)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [[10, 10], [20, 30, 35]])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_zeros(shape, dtype):
    """
    Feature: Test torch.zeros
    Description: Test empty with dtype inputs
    Expectation: The result is correct
    """

    cpu_output0 = zeros_func(shape, dtype)
    op_func_compiled = get_zeros_op_func_compiled()
    npu_output = op_func_compiled(shape, dtype)
    npu_output0 = npu_output.detach().cpu()
    AssertRtolEqual(cpu_output0, npu_output0)
    assert npu_output.device.type == 'npu', f"device should be npu, but got {npu_output.device.type}"
