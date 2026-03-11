"""Tests for torch.split operation."""
import pytest
import torch

from ms_inferrt.torch.fx_mlir_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(input_self_tensor, split_size, dim=0):
    """Reference implementation of split."""
    return input_self_tensor.split(split_size, dim)


def get_op_func_compiled():
    """Get compiled split function."""
    def custom_op_func(input_self_tensor, split_size, dim=0):
        return input_self_tensor.split(split_size, dim)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [[128, 4096], [32, 1024]])
def test_split_tensor(shape):
    """
    Feature: Test split
    Description: Test split with bf16 inputs
    Expectation: The result is correct
    """

    self_tensor = torch.rand(shape, dtype=torch.bfloat16)
    dim = 0
    self_tensor_npu = self_tensor.npu()
    split_size = shape[0] // 2

    cpu_output0, cpu_output1 = op_func(self_tensor, split_size, dim=dim)
    op_func_compiled = get_op_func_compiled()
    npu_output0, npu_output1 = op_func_compiled(self_tensor_npu, split_size, dim=dim)
    npu_output_opt0 = npu_output0.detach().cpu()
    npu_output_opt1 = npu_output1.detach().cpu()
    AssertRtolEqual(cpu_output0, npu_output_opt0)
    AssertRtolEqual(cpu_output1, npu_output_opt1)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [[128, 4096], [32, 1024]])
def test_split_with_size(shape):
    """
    Feature: Test split
    Description: Test split with bf16 inputs
    Expectation: The result is correct
    """

    self_tensor = torch.rand(shape, dtype=torch.bfloat16)
    dim = 0
    self_tensor_npu = self_tensor.npu()
    split_size = [shape[0] // 2, shape[0] // 2]

    cpu_output0, cpu_output1 = op_func(self_tensor, split_size, dim=dim)
    op_func_compiled = get_op_func_compiled()
    npu_output0, npu_output1 = op_func_compiled(self_tensor_npu, split_size, dim=dim)
    npu_output_opt0 = npu_output0.detach().cpu()
    npu_output_opt1 = npu_output1.detach().cpu()
    AssertRtolEqual(cpu_output0, npu_output_opt0)
    AssertRtolEqual(cpu_output1, npu_output_opt1)
