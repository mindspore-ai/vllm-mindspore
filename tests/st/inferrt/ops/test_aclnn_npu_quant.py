# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for npu_quantize operation.

This module contains tests for the npu_quantize operation
on Ascend NPU devices with different backends and data types.
"""
import pytest
import torch
import torch_npu
from ms_inferrt.torch.fx_backend import backend as fx_backend
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
def test_npu_quantize_per_channel_axis1(backend, dtype):
    """
    Feature: Check npu_quantize op with per-channel quantization (axis=1)
    Description: Test per-channel quantization along axis=1 with qint8 output
    Expectation: The result matches torch_npu native implementation
    """

    def npu_quantize_func(x, scales, zero_points):
        # Use div_mode=False to trigger AscendQuantV3 path
        return torch_npu.npu_quantize(x, scales, zero_points, torch.qint8, 1, div_mode=False)

    compiled_func = torch.compile(npu_quantize_func, backend=backend)

    # Generate test data: shape (batch, channels) = (16, 128)
    x = torch.randn([16, 128], dtype=dtype).npu()
    # Per-channel scales and zero_points for axis=1
    scales = torch.randn([128], dtype=dtype).npu()
    zero_points = torch.randn([128], dtype=dtype).npu()

    # Execute compiled function
    output = compiled_func(x, scales, zero_points)

    # Reference implementation
    expected_output = torch_npu.npu_quantize(x, scales, zero_points, torch.qint8, 1, div_mode=False)

    assert torch.allclose(output.cpu().float(), expected_output.cpu().float(), rtol=0.01, atol=0.01)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_npu_quantize_per_tensor(backend):
    """
    Feature: Check npu_quantize op with per-tensor quantization (axis=-1)
    Description: Test per-tensor quantization with qint8 output
    Expectation: The result matches torch_npu native implementation
    """

    def npu_quantize_func(x, scales, zero_points):
        return torch_npu.npu_quantize(x, scales, zero_points, torch.qint8, -1, div_mode=False)

    compiled_func = torch.compile(npu_quantize_func, backend=backend)

    # Generate test data
    x = torch.randn([16, 128], dtype=torch.float16).npu()
    # Per-tensor scales and zero_points (shape [1])
    scales = torch.randn([1], dtype=torch.float16).npu()
    zero_points = torch.randn([1], dtype=torch.float16).npu()

    output = compiled_func(x, scales, zero_points)
    expected_output = torch_npu.npu_quantize(x, scales, zero_points, torch.qint8, -1, div_mode=False)

    assert torch.allclose(output.cpu().float(), expected_output.cpu().float(), rtol=0.01, atol=0.01)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_npu_quantize_per_head(backend):
    """
    Feature: Check npu_quantize op with per-head quantization (axis=-2)
    Description: Test per-head quantization where scales shape is (heads, 1)
    Expectation: The result matches torch_npu native implementation
    """

    def npu_quantize_func(x, scales, zero_points):
        return torch_npu.npu_quantize(x, scales, zero_points, torch.qint8, -2, div_mode=False)

    compiled_func = torch.compile(npu_quantize_func, backend=backend)

    # Shape: (heads, hidden_dim) = (16, 128)
    x = torch.randn([16, 128], dtype=torch.float16).npu()
    # Per-head scales: shape (16, 1) for axis=-2
    scales = torch.randn([16, 1], dtype=torch.float16).npu()
    zero_points = torch.randn([16, 1], dtype=torch.float16).npu()

    output = compiled_func(x, scales, zero_points)
    expected_output = torch_npu.npu_quantize(x, scales, zero_points, torch.qint8, -2, div_mode=False)

    assert torch.allclose(output.cpu().float(), expected_output.cpu().float(), rtol=0.01, atol=0.01)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_npu_quantize_without_zero_points(backend):
    """
    Feature: Check npu_quantize op with None zero_points
    Description: Test quantization without zero_points (offset)
    Expectation: The result matches torch_npu native implementation
    """

    def npu_quantize_func(x, scales):
        return torch_npu.npu_quantize(x, scales, None, torch.qint8, -1, div_mode=False)

    compiled_func = torch.compile(npu_quantize_func, backend=backend)

    x = torch.randn([32, 64], dtype=torch.float16).npu()
    scales = torch.randn([64], dtype=torch.float16).npu()

    output = compiled_func(x, scales)
    expected_output = torch_npu.npu_quantize(x, scales, None, torch.qint8, -1, div_mode=False)

    assert torch.allclose(output.cpu().float(), expected_output.cpu().float(), rtol=0.01, atol=0.01)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
@pytest.mark.parametrize("backend", (fx_backend,))
@pytest.mark.parametrize("dtype", (torch.quint4x2,))
def test_npu_quantize_different_dtypes(backend, dtype):
    """
    Feature: Check npu_quantize op with different output dtypes
    Description: Test quint8 and qint32 output data types
    Expectation: The result matches torch_npu native implementation
    """

    def npu_quantize_func(x, scales, zero_points):
        return torch_npu.npu_quantize(x, scales, zero_points, dtype, 1, div_mode=False)

    compiled_func = torch.compile(npu_quantize_func, backend=backend)

    x = torch.randn([8, 32], dtype=torch.float32).npu()
    scales = torch.randn([32], dtype=torch.float32).npu()
    zero_points = torch.randn([32], dtype=torch.float32).npu()

    output = compiled_func(x, scales, zero_points)
    expected_output = torch_npu.npu_quantize(x, scales, zero_points, dtype, 1, div_mode=False)

    assert torch.allclose(output.cpu().float(), expected_output.cpu().float(), rtol=0.01, atol=0.01)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_npu_quantize_high_dim(backend):
    """
    Feature: Check npu_quantize op with high dimensional input
    Description: Test quantization on 4D input tensor with axis=3 (per-channel on last dim)
    Expectation: The result matches torch_npu native implementation
    """

    def npu_quantize_func(x, scales, zero_points):
        return torch_npu.npu_quantize(x, scales, zero_points, torch.qint8, 3, div_mode=False)

    compiled_func = torch.compile(npu_quantize_func, backend=backend)

    # 4D tensor: (N, C, H, W) = (4, 8, 16, 32)
    x = torch.randn([4, 8, 16, 32], dtype=torch.float16).npu()

    # Per-channel scales for axis=3 (W dimension)
    scales = torch.randn([32], dtype=torch.float16).npu()
    zero_points = torch.randn([32], dtype=torch.float16).npu()

    output = compiled_func(x, scales, zero_points)
    expected_output = torch_npu.npu_quantize(x, scales, zero_points, torch.qint8, 3, div_mode=False)

    assert torch.allclose(output.cpu().float(), expected_output.cpu().float(), rtol=0.01, atol=0.01)
