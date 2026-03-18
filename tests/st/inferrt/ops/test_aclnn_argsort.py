"""Tests for aclnn argsort operation."""
import numpy as np
import pytest
import torch

from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def argsort_default(x):
    """argsort with default parameters (dim=-1, descending=False)"""
    return torch.argsort(x)


def argsort_dim0_descending(x):
    """argsort along dim 0 with descending=True"""
    return torch.argsort(x, dim=0, descending=True)


def argsort_stable_descending(x):
    """argsort with stable=True and descending=True"""
    return torch.argsort(x, stable=True, dim=0, descending=True)


def argsort_method_stable_descending(x):
    """Tensor.argsort method with stable=True and descending=True."""
    return x.argsort(stable=True, dim=0, descending=True)


def argsort_forward(dtype, shape, op_func, compiled_func):
    """
    argsort forward function
    Args:
        dtype: The data type of the input.
        shape: The shape of the input tensor.
        op_func: The original op function.
        compiled_func: The compiled op function.
    """
    if np.issubdtype(dtype, np.integer):
        cpu_input = np.random.randint(-100, 100, shape).astype(dtype)
    else:
        cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)

    npu_input = torch.from_numpy(cpu_input).npu()

    ori_output = op_func(npu_input).detach().cpu().numpy()
    npu_output = compiled_func(npu_input).detach().cpu().numpy()

    # argsort returns indices, so we should have exact match
    AssertRtolEqual(ori_output, npu_output, 0)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(128, 256, 64)])
@pytest.mark.parametrize("op_func", [
    argsort_default, argsort_dim0_descending, argsort_stable_descending, argsort_method_stable_descending
])
# pylint: disable=redefined-outer-name
def test_argsort_fp32_3d(shape, op_func):
    """
    Feature: Test aclnn argsort on 3D tensor
    Description: Test aclnn argsort with fp32 3D inputs
    Expectation: The result is correct
    """
    compiled_op = torch.compile(op_func, backend=backend)
    argsort_forward(np.float32, shape, op_func, compiled_op)
