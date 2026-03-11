"""Tests for aclnn flatten operation."""
import pytest
import torch

from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(input_self_tensor, start_dim = 0, end_idx = -1):
    return input_self_tensor.flatten(start_dim, end_idx)

def get_op_func_compiled():
    def custom_op_func(input_self_tensor, start_dim = 0, end_idx = -1):
        return input_self_tensor.flatten(start_dim, end_idx)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [[4, 5, 6, 7, 8], [7, 6, 8, 10]])
@pytest.mark.parametrize("start_idx", [0, 1])
@pytest.mark.parametrize("end_idx", [-1, 3])
def test_flatten(shape, start_idx, end_idx):
    """
    Feature: Test aclnn flatten
    Description: Test aclnn flatten with bf16 inputs
    Expectation: The result is correct
    """

    self_tensor = torch.rand(shape, dtype=torch.bfloat16)
    self_tensor_npu = self_tensor.npu()

    cpu_output0  = op_func(self_tensor, start_dim=start_idx, end_idx=end_idx)
    op_func_compiled = get_op_func_compiled()
    npu_output0 = op_func_compiled(self_tensor_npu, start_dim=start_idx, end_idx=end_idx)
    npu_output_opt0 = npu_output0.detach().cpu()
    AssertRtolEqual(cpu_output0, npu_output_opt0)
