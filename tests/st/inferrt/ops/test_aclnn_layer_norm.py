"""Tests for torch.nn.functional.layer_norm operation."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ms_inferrt.torch.fx_mlir_backend import backend as mlir_backend
from ms_inferrt.torch.fx_backend import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(x, normalized_shape, weight, bias, eps=1e-5):
    """Reference implementation of layer_norm using PyTorch eager mode."""
    return F.layer_norm(x, normalized_shape, weight, bias, eps)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend, mlir_backend))
def test_layer_norm(backend):
    """
    Feature: Test aclnn layer_norm
    Description: Test aclnn layer_norm with fp32 inputs and different backends
    Expectation: The result is correct
    """

    def custom_op_func(x, normalized_shape, weight, bias, eps=1e-5):
        return F.layer_norm(x, normalized_shape, weight, bias, eps)

    op_func_compiled = torch.compile(custom_op_func, backend=backend)

    normalized_shape = (4,)

    cpu_input = np.random.randn(2, 3, 4).astype(np.float32)
    cpu_weight = np.random.randn(4).astype(np.float32)
    cpu_bias = np.random.randn(4).astype(np.float32)

    cpu_tensor_input = torch.from_numpy(cpu_input)
    cpu_tensor_weight = torch.from_numpy(cpu_weight)
    cpu_tensor_bias = torch.from_numpy(cpu_bias)

    npu_input = cpu_tensor_input.npu()
    npu_weight = cpu_tensor_weight.npu()
    npu_bias = cpu_tensor_bias.npu()

    cpu_output = op_func(cpu_tensor_input, normalized_shape, cpu_tensor_weight, cpu_tensor_bias)

    npu_output = op_func_compiled(npu_input, normalized_shape, npu_weight, npu_bias)

    cpu_output_np = cpu_output.detach().numpy()
    npu_output_np = npu_output.detach().cpu().numpy()

    AssertRtolEqual(cpu_output_np, npu_output_np)
