"""Tests for clone operation with dynamic shapes."""

import numpy as np
import pytest
import torch

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def clone_dynamic_op(x):
    b = x.size(0)
    return torch.clone(x[:b])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8,), (4, 8), (16, 32), (2, 4, 8), (1, 8, 16, 32)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_clone_dynamic(shape, dtype):
    """
Feature: Test clone with dynamic shapes.
    Description: Test torch.clone with dynamic input slicing.
    Expectation: The result matches eager mode.
    """
    compiled_op = torch.compile(clone_dynamic_op, backend=backend)
    prec = 0.001 if dtype == np.float16 else 0.0001
    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    npu_input = torch.from_numpy(cpu_input).npu()
    cpu_output = clone_dynamic_op(torch.from_numpy(cpu_input)).detach().numpy()
    npu_output = compiled_op(npu_input).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output, prec)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8, 16), (2, 4, 8), (1, 8, 16, 32)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_clone_static(shape, dtype):
    """
Feature: Test clone with static shapes.
    Description: Test torch.clone with fixed shapes.
    Expectation: The result matches eager mode.
    """
    def clone_static_op(x):
        return torch.clone(x)

    compiled_op = torch.compile(clone_static_op, backend=backend)
    prec = 0.001 if dtype == np.float16 else 0.0001
    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    npu_input = torch.from_numpy(cpu_input).npu()
    cpu_output = clone_static_op(torch.from_numpy(cpu_input)).detach().numpy()
    npu_output = compiled_op(npu_input).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output, prec)
