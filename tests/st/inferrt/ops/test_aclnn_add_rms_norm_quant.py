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
Tests for npu_add_rms_norm_quant operation.

This module contains tests for the npu_add_rms_norm_quant operation
on Ascend NPU devices with different backends and data types.
"""

import pytest
import torch
import torch_npu
from ms_inferrt.torch.fx_backend import backend as fx_backend
from ms_inferrt.torch.fx_mlir_backend import backend as mlir_backend
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend, mlir_backend))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_npu_add_rms_norm_quant(backend, dtype):
    """
    Feature: Check npu_add_rms_norm_quant op launch
    Description: Check npu_add_rms_norm_quant op launch with float16 and bfloat16
    Expectation: The result is correct
    """
    def npu_add_rms_norm_quant_func(x1, x2, gamma, scales1, zero_points1):
        return torch_npu.npu_add_rms_norm_quant(x1, x2, gamma, scales1, zero_points1)

    compiled_func = torch.compile(npu_add_rms_norm_quant_func, backend=backend)

    x1 = torch.randn([2, 16], dtype=dtype).npu()
    x2 = torch.randn([2, 16], dtype=dtype).npu()
    gamma = torch.randn([16, ], dtype=dtype).npu()
    scales1 = torch.randn([16, ], dtype=dtype).npu()

    if dtype == torch.float16:
        zero_points1 = torch.randint(-10, 10, [16, ], dtype=torch.int32).npu()
    else:
        zero_points1 = torch.randn([16, ], dtype=dtype).npu()

    y1, _, x_out = compiled_func(x1, x2, gamma, scales1, zero_points1)
    expected_y1, _, expected_x_out = torch_npu.npu_add_rms_norm_quant(
        x1, x2, gamma, scales1, zero_points1
    )

    assert torch.allclose(y1, expected_y1)
    assert torch.allclose(x_out, expected_x_out)
