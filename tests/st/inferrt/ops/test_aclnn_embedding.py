"""Tests for aclnn embedding operation."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ms_inferrt.torch import fx_mlir_backend as backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(indices, weight):
    out = F.embedding(indices, weight)
    return out


def get_op_func_compiled():
    def custom_op_func(indices, weight):
        return F.embedding(indices, weight)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
def test_embedding(dtype):
    """
    Feature: Test aclnn embedding
    Description: Test aclnn embedding with fp32/fp16 inputs
    Expectation: The result is correct
    """
    cpu_weight_torch = torch.from_numpy(np.random.rand(10, 3).astype(np.float32)).to(dtype)
    cpu_indices = torch.from_numpy(np.array([[1, 2, 4, 5], [4, 3, 2, 9]]))

    npu_weight_torch = cpu_weight_torch.npu()
    npu_indices = cpu_indices.npu()

    cpu_output0 = op_func(cpu_indices, cpu_weight_torch)
    op_func_compiled = get_op_func_compiled()
    npu_output0 = [npu_output.detach().cpu().numpy() for npu_output in op_func_compiled(npu_indices, npu_weight_torch)]
    AssertRtolEqual(cpu_output0, npu_output0)

    cpu_weight_torch = torch.from_numpy(np.random.rand(20, 4).astype(np.float32)).to(dtype)
    cpu_indices = torch.from_numpy(np.array([[1, 2, 4, 5], [4, 3, 2, 9]]))

    npu_weight_torch = cpu_weight_torch.npu()
    npu_indices = cpu_indices.npu()

    cpu_output0 = op_func(cpu_indices, cpu_weight_torch)
    op_func_compiled = get_op_func_compiled()
    npu_output0 = [npu_output.detach().cpu().numpy() for npu_output in op_func_compiled(npu_indices, npu_weight_torch)]
    AssertRtolEqual(cpu_output0, npu_output0)
