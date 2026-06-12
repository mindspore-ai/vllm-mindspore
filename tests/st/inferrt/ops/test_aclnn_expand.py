"""Tests for aclnn expand operation."""
import numpy as np
import pytest
import torch

from ms_inferrt.torch import fx_mlir_backend as backend
from ms_inferrt.torch.fx_backend import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def expand_op(x, y):
    """op function for expand with dynamic size based on x.size"""
    # Use size(int) so that aten.size.int is in the graph
    b = x.size(0)
    target_shape = (b, 16)
    return torch.broadcast_to(y, target_shape)


def get_op_func_compiled():
    return torch.compile(expand_op, backend=backend)


# 去掉 shape 的 parametrize，改在内部循环
@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(2, 8), (4, 8)])
def test_expand(shape):
    """
    Feature: Test aclnn expand
    Description: Test aclnn expand with dynamic size based on x.size
    Expectation: The result is correct
    """


    compile_op = get_op_func_compiled()
    prec = 1e-4

    cpu_input0 = np.random.uniform(-1, 1, shape).astype(np.float32)
    cpu_input1 = np.random.uniform(-1, 1, (1, 1)).astype(np.float32)
    cpu_tensor0 = torch.from_numpy(cpu_input0)
    cpu_tensor1 = torch.from_numpy(cpu_input1)
    npu_tensor0 = torch.from_numpy(cpu_input0).npu()
    npu_tensor1 = torch.from_numpy(cpu_input1).npu()

    cpu_output = expand_op(cpu_tensor0, cpu_tensor1).detach().cpu().numpy()
    npu_output = compile_op(npu_tensor0, npu_tensor1).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output, prec)


def expand_with_neg1_op(x):
    """Test expand with -1 in size via fx_backend (covers schema unpack, string 'expand' map, -1 handling)."""
    return x.expand(1, -1, -1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(1, 1, 16), (1, 4, 8), (1, 1, 32)])
def test_expand_neg1(shape):
    """
    Feature: Test expand with -1 size via fx_backend
    Description: Test aten.expand with unpacked args and -1 size
    Expectation: Correct result, no schema error, no aclnnExpand -1 mismatch
    """
    compile_op = torch.compile(expand_with_neg1_op, backend=fx_backend)
    prec = 1e-4

    cpu_input = np.random.uniform(-1, 1, shape).astype(np.float32)
    cpu_tensor = torch.from_numpy(cpu_input)
    npu_tensor = torch.from_numpy(cpu_input).npu()

    cpu_output = expand_with_neg1_op(cpu_tensor).detach().cpu().numpy()
    npu_output = compile_op(npu_tensor).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output, prec)

def aten_expand_dynamic_op(x):
    b = x.size(0)
    ndim = x.dim()
    sizes = [b] + [-1] * ndim
    return torch.ops.aten.expand.default(x[:b].unsqueeze(0), sizes)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8,), (4, 8), (16, 32), (2, 4, 8)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_aten_expand_dynamic(shape, dtype):
    """
Feature: Test aten expand with dynamic shapes.
    Description: Test aten.expand.default with various shapes.
    Expectation: The result matches eager mode.
    """
    compiled_op = torch.compile(aten_expand_dynamic_op, backend=backend)
    prec = 0.001 if dtype == np.float16 else 0.0001
    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    npu_input = torch.from_numpy(cpu_input).npu()
    cpu_output = aten_expand_dynamic_op(torch.from_numpy(cpu_input)).detach().numpy()
    npu_output = compiled_op(npu_input).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output, prec)

def expand_dynamic_op(x, y):
    return y.expand(x.size(0), -1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8,), (4, 8), (16, 32), (2, 4, 8)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_expand_dynamic(shape, dtype):
    """
Feature: Test expand with dynamic shapes.
    Description: Test tensor.expand with dynamic target shape.
    Expectation: The result matches eager mode.
    """
    compiled_op = torch.compile(expand_dynamic_op, backend=backend)
    prec = 0.001 if dtype == np.float16 else 0.0001
    cpu_input0 = np.random.uniform(-1, 1, shape).astype(dtype)
    cpu_input1 = np.random.uniform(-1, 1, (1, 1)).astype(dtype)
    npu_input0 = torch.from_numpy(cpu_input0).npu()
    npu_input1 = torch.from_numpy(cpu_input1).npu()
    cpu_output = expand_dynamic_op(torch.from_numpy(cpu_input0), torch.from_numpy(cpu_input1)).detach().numpy()
    npu_output = compiled_op(npu_input0, npu_input1).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output, prec)

def expand_neg1_dynamic_op(x):
    b = x.size(0)
    return x[:b].expand(1, -1, -1)



@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(1, 1, 16), (1, 4, 8), (1, 1, 32)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_expand_neg1_dynamic(shape, dtype):
    """
    Feature: Test expand with -1 dimensions and dynamic shapes.
    Description: Test tensor.expand(1, -1, -1) with dynamic slicing.
    Expectation: The result matches eager mode.
    """
    compiled_op = torch.compile(expand_neg1_dynamic_op, backend=fx_backend)
    prec = 0.001 if dtype == np.float16 else 0.0001
    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)
    npu_input = torch.from_numpy(cpu_input).npu()
    cpu_output = expand_neg1_dynamic_op(torch.from_numpy(cpu_input)).detach().numpy()
    npu_output = compiled_op(npu_input).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output, prec)
