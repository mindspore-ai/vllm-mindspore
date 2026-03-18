# Copyright 2026 Huawei Technologies Co., Ltd
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
Tests for npu_dynamic_quant operation.

This module contains tests for the npu_dynamic_quant operation
on Ascend NPU devices with different backends and data types.
"""

import pytest
import numpy as np
import torch
import torch_npu
from ms_inferrt.torch.fx_backend import backend as fx_backend
from ms_inferrt.torch.fx_mlir_backend import backend as mlir_backend
from tests.mark_utils import arg_mark


def generate_input(input_shape, dtype="float16", use_smooth=False, group_num=1):  # pylint: disable=missing-function-docstring
    date_type = torch.float16 if dtype == "float16" else torch.bfloat16
    input_tensor = torch.randn(input_shape, dtype=date_type)
    group_index = None
    smooth_scales = None
    if group_num > 1:
        smooth_scales = torch.randn(group_num, input_shape[-1], dtype=date_type)
        row_num = input_tensor.numel() // input_tensor.shape[-1]
        group_index_list = []
        for _ in range(group_num):
            group_index_list.append(np.random.randint(0, row_num))
        group_index_list = sorted(group_index_list)
        group_index_list[-1] = row_num
        group_index = torch.tensor(group_index_list).to(torch.int32)
    elif use_smooth:
        smooth_scales = torch.randn(input_shape[-1], dtype=date_type)
    return input_tensor, smooth_scales, group_index


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend, mlir_backend))
@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_npu_dynamic_quant(backend, dtype):
    """
    Feature: Check npu_dynamic_quant op launch
    Description: Check npu_dynamic_quant op launch with fp16/bf16, default int8 dst_type
    Expectation: The result is correct
    """

    def npu_dynamic_quant_func(x):
        return torch_npu.npu_dynamic_quant(x)

    compiled_func = torch.compile(npu_dynamic_quant_func, backend=backend)

    input_tensor, _, _ = generate_input(input_shape=[2, 32, 256], dtype=dtype, use_smooth=False, group_num=1)
    input_tensor = input_tensor.npu()

    output, scale = compiled_func(input_tensor)
    expected_output, expected_scale = torch_npu.npu_dynamic_quant(input_tensor)

    assert torch.allclose(output, expected_output)
    assert torch.allclose(scale, expected_scale)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend, mlir_backend))
@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_npu_dynamic_quant_smooth_group(backend, dtype):
    """
    Feature: Check npu_dynamic_quant op launch with smooth and group
    Description: Check npu_dynamic_quant op launch with smooth_scales and group_index
    Expectation: The result is correct
    """

    def npu_dynamic_quant_func(x, smooth_scales, group_index):
        return torch_npu.npu_dynamic_quant(x, smooth_scales=smooth_scales, group_index=group_index, dst_type=torch.int8)

    compiled_func = torch.compile(npu_dynamic_quant_func, backend=backend)

    input_tensor, smooth_scales, group_index = generate_input(input_shape=[2, 32, 256], dtype=dtype, use_smooth=True,
                                                              group_num=64)
    input_tensor, smooth_scales, group_index = input_tensor.npu(), smooth_scales.npu(), group_index.npu()

    output, scale = compiled_func(input_tensor, smooth_scales, group_index)
    expected_output, expected_scale = torch_npu.npu_dynamic_quant(input_tensor, smooth_scales=smooth_scales,
                                                                  group_index=group_index, dst_type=torch.int8)

    assert torch.allclose(output, expected_output)
    assert torch.allclose(scale, expected_scale)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_npu_dynamic_quant_int4(backend, dtype):
    """
    Feature: Check npu_dynamic_quant op launch with quint4x2 output
    Description: Check npu_dynamic_quant op launch with torch.quint4x2
    Expectation: The result is correct
    """

    def npu_dynamic_quant_func(x):
        return torch_npu.npu_dynamic_quant(x, dst_type=torch.quint4x2)

    compiled_func = torch.compile(npu_dynamic_quant_func, backend=backend)

    input_tensor, _, _ = generate_input(input_shape=[2, 32, 256], dtype=dtype, use_smooth=False, group_num=1)
    input_tensor = input_tensor.npu()

    output, scale = compiled_func(input_tensor)
    expected_output, expected_scale = torch_npu.npu_dynamic_quant(input_tensor, dst_type=torch.quint4x2)

    assert torch.allclose(output.view(torch.int32), expected_output.view(torch.int32))
    assert torch.allclose(scale, expected_scale)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_npu_dynamic_quant_int4_smooth_group(backend, dtype):
    """
    Feature: Check npu_dynamic_quant op launch with quint4x2 output, smooth and group
    Description: Check npu_dynamic_quant op launch with smooth_scales, group_index and torch.quint4x2
    Expectation: The result is correct
    """

    def npu_dynamic_quant_func(x, smooth_scales, group_index):
        return torch_npu.npu_dynamic_quant(x, smooth_scales=smooth_scales, group_index=group_index,
                                           dst_type=torch.quint4x2)

    compiled_func = torch.compile(npu_dynamic_quant_func, backend=backend)

    input_tensor, smooth_scales, group_index = generate_input(input_shape=[2, 32, 256], dtype=dtype, use_smooth=True,
                                                              group_num=64)
    input_tensor, smooth_scales, group_index = input_tensor.npu(), smooth_scales.npu(), group_index.npu()

    output, scale = compiled_func(input_tensor, smooth_scales, group_index)
    expected_output, expected_scale = torch_npu.npu_dynamic_quant(input_tensor, smooth_scales=smooth_scales,
                                                                  group_index=group_index, dst_type=torch.quint4x2)

    assert torch.allclose(output.view(torch.int32), expected_output.view(torch.int32))
    assert torch.allclose(scale, expected_scale)
