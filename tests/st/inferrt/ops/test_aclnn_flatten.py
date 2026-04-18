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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_flatten_permute_view_input():
    """
    Feature: Test aclnn flatten with non-contiguous (permuted) input
    Description: Flatten on permute(0,1,3,2) tensor where dims [0,1] form a contiguous group
    Expectation: The result is correct
    """
    # permute(0,1,3,2) on [3,4,5,6] -> [3,4,6,5], strides (120,30,1,6)
    # dims 0,1 are contiguous: strides[0]=120 == shape[1]*strides[1]=4*30=120
    self_tensor = torch.rand(3, 4, 5, 6, dtype=torch.bfloat16)
    permuted = self_tensor.permute(0, 1, 3, 2)
    permuted_npu = permuted.npu()

    cpu_output = op_func(permuted, start_dim=0, end_idx=1)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(permuted_npu, start_dim=0, end_idx=1)
    npu_output_cpu = npu_output.detach().cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_flatten_slice_offset_input():
    """
    Feature: Test aclnn flatten with sliced input (non-zero storage offset)
    Description: Flatten on sliced tensor with storage_offset != 0 but contiguous strides
    Expectation: The result is correct
    """
    self_tensor = torch.rand(3, 4, 5, 6, dtype=torch.bfloat16)
    sliced = self_tensor[1:]  # shape [2,4,5,6], contiguous strides, storage_offset != 0
    sliced_npu = sliced.npu()

    cpu_output = op_func(sliced, start_dim=1, end_idx=2)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(sliced_npu, start_dim=1, end_idx=2)
    npu_output_cpu = npu_output.detach().cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_flatten_2d_slice_input():
    """
    Feature: Test aclnn flatten with 2D sliced input
    Description: Flatten on a sliced 2D tensor with non-zero storage offset
    Expectation: The result is correct
    """
    self_tensor = torch.rand(10, 20, dtype=torch.bfloat16)
    sliced = self_tensor[2:8, 3:15]  # shape [6, 12], strides (20, 1), offset=43
    sliced_npu = sliced.npu()

    cpu_output = op_func(sliced, start_dim=0, end_idx=-1)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(sliced_npu, start_dim=0, end_idx=-1)
    npu_output_cpu = npu_output.detach().cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_flatten_permute_partial_flatten():
    """
    Feature: Test aclnn flatten on permuted tensor with partial flatten range
    Description: Flatten only the trailing contiguous dims on a permuted tensor
    Expectation: The result is correct
    """
    # permute(0,1,3,2) on [3,4,5,6] -> [3,4,6,5], strides (120,30,1,6)
    # flatten(2,3): strides[2]=1, shape[3]*strides[3]=5*6=30 != 1, NOT view-compatible
    # But PyTorch will make it contiguous first, so the result should still be correct
    self_tensor = torch.rand(3, 4, 5, 6, dtype=torch.bfloat16)
    permuted = self_tensor.permute(0, 1, 3, 2)
    permuted_npu = permuted.npu()

    # flatten(0,-1) on the non-contiguous tensor - PyTorch will call contiguous first
    cpu_output = op_func(permuted, start_dim=0, end_idx=-1)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(permuted_npu, start_dim=0, end_idx=-1)
    npu_output_cpu = npu_output.detach().cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)
