"""Tests for bmm operation with dynamic shapes."""

import numpy as np
import pytest
import torch

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def bmm_dynamic_op(x1, x2):
    b = x1.size(0)
    return torch.bmm(x1[:b], x2[:b])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("batch,n,m,k", [(1, 4, 8, 4), (2, 16, 32, 16), (4, 8, 16, 8), (8, 4, 8, 4), (4, 32, 64, 32)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_bmm_dynamic(batch, n, m, k, dtype):
    """
Feature: Test bmm with dynamic shapes.
    Description: Test torch.bmm with dynamic batch slicing.
    Expectation: The result matches eager mode.
    """
    compiled_op = torch.compile(bmm_dynamic_op, backend=backend)
    prec = 0.001 if dtype == np.float16 else 0.0001
    shape1 = (batch, n, m)
    shape2 = (batch, m, k)
    cpu_input0 = np.random.uniform(-1, 1, shape1).astype(dtype)
    cpu_input1 = np.random.uniform(-1, 1, shape2).astype(dtype)
    npu_input0 = torch.from_numpy(cpu_input0).npu()
    npu_input1 = torch.from_numpy(cpu_input1).npu()
    cpu_output = bmm_dynamic_op(torch.from_numpy(cpu_input0), torch.from_numpy(cpu_input1)).detach().numpy()
    npu_output = compiled_op(npu_input0, npu_input1).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output, prec)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("batch,n,m,k", [(1, 4, 8, 4), (2, 16, 32, 16), (4, 32, 64, 32)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_bmm_static(batch, n, m, k, dtype):
    """
Feature: Test bmm with static shapes.
    Description: Test torch.bmm with fixed shapes.
    Expectation: The result matches eager mode.
    """
    def bmm_static_op(x1, x2):
        return torch.bmm(x1, x2)

    compiled_op = torch.compile(bmm_static_op, backend=backend)
    prec = 0.001 if dtype == np.float16 else 0.0001
    shape1 = (batch, n, m)
    shape2 = (batch, m, k)
    cpu_input0 = np.random.uniform(-1, 1, shape1).astype(dtype)
    cpu_input1 = np.random.uniform(-1, 1, shape2).astype(dtype)
    npu_input0 = torch.from_numpy(cpu_input0).npu()
    npu_input1 = torch.from_numpy(cpu_input1).npu()
    cpu_output = bmm_static_op(torch.from_numpy(cpu_input0), torch.from_numpy(cpu_input1)).detach().numpy()
    npu_output = compiled_op(npu_input0, npu_input1).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output, prec)
