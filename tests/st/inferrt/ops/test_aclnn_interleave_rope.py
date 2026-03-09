"""
Tests for npu_interleave_rope operation.

This module contains tests for the npu_interleave_rope operation
on Ascend NPU devices with different backends and data types.
"""

import pytest
import torch
import torch_npu
from ms_inferrt.torch.fx_backend import backend as fx_backend
from ms_inferrt.torch.fx_mlir_backend import backend as mlir_backend
from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend, mlir_backend))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_interleave_rope(backend, dtype):
    """
    Feature: Check npu_interleave_rope op launch
    Description: Check npu_interleave_rope op launch with fp16/bf16
    Expectation: The result is correct
    """
    def npu_interleave_rope_func(x, cos, sin):
        return torch_npu.npu_interleave_rope(x, cos, sin)

    compiled_func = torch.compile(npu_interleave_rope_func, backend=backend)

    # Use shapes from example
    x_shape = (32, 32, 1, 64)
    cos_shape = (32, 1, 1, 64)

    # Generate CPU tensors
    x_cpu = torch.randn(x_shape, dtype=dtype)
    cos_cpu = torch.randn(cos_shape, dtype=dtype)
    sin_cpu = torch.randn(cos_shape, dtype=dtype)

    # Clone to NPU
    x_npu = x_cpu.clone().npu()
    cos_npu = cos_cpu.clone().npu()
    sin_npu = sin_cpu.clone().npu()

    # Eager mode execution
    expected_output = torch_npu.npu_interleave_rope(x_npu, cos_npu, sin_npu)

    # Graph mode execution
    output = compiled_func(x_npu, cos_npu, sin_npu)

    # Compare results
    AssertRtolEqual(expected_output, output)
