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
Test for quantized matrix multiplication operator.

This module contains tests for the NPU quantized matrix multiplication
operation using FX backend, including W8A8 and FP16 formats.
"""

import pytest
import torch
import torch_npu

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual
from ms_inferrt.torch import backend


def op_func(x1, x2, scale):
    """
    quant_matmul forward function using torch.matmul
    Args:
        x1: The first input tensor.
        x2: The second input tensor.
        scale: The scale tensor.
        output_dtype: The output dtype (optional).
    """
    return torch.matmul(x1, x2) * scale


def get_op_func_compiled():
    """
    Get compiled quant_matmul function
    """
    def custom_op_func(x1, x2, scale, output_dtype=None):
        return torch_npu.npu_quant_matmul(x1, x2, scale, output_dtype=output_dtype)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("batch_size", [4096, 8192])
@pytest.mark.parametrize("in_features", [160, 320])
@pytest.mark.parametrize("out_features", [1280, 2560])
def test_quant_matmul_a8w8(batch_size, in_features, out_features, dtype):
    """
    Feature: Test aclnn quant_matmul
    Description: Test aclnn quant_matmul with W8A8
    Expectation: The result is correct
    """
    x1 = torch.randint(-5, 5, (batch_size, in_features), dtype=torch.int8)
    x2 = torch.randint(-5, 5, (in_features, out_features), dtype=torch.int8)
    x1_npu = x1.clone().npu()
    x2_npu = x2.clone().npu()
    scale = torch.randn(1, dtype=torch.float32)
    if dtype == torch.bfloat16:
        scale_quant = scale.npu()
    else:
        scale_quant = torch_npu.npu_trans_quant_param(scale.npu(), None)

    expected_output = op_func(x1.to(torch.int32), x2.to(torch.int32), scale)
    op_func_compiled = get_op_func_compiled()

    custom_output = op_func_compiled(x1_npu, x2_npu, scale_quant, output_dtype=dtype)
    AssertRtolEqual(expected_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.01)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("batch_size", [4096, 8192])
@pytest.mark.parametrize("in_features", [160, 320])
@pytest.mark.parametrize("out_features", [1280, 2560])
def test_quant_matmul_fp16_nz(batch_size, in_features, out_features, dtype):
    """
    Feature: Test aclnn quant_matmul
    Description: Test aclnn quant_matmul with W8A8 and NZ format
    Expectation: The result is correct
    """
    x1 = torch.randint(-5, 5, (batch_size, in_features), dtype=torch.int8)
    x2 = torch.randint(-5, 5, (in_features, out_features), dtype=torch.int8)
    x1_npu = x1.clone().npu()
    x2_npu = x2.clone().npu()
    x2_nz = torch_npu.npu_format_cast(x2_npu.contiguous(), 29)
    scale = torch.randn(1, dtype=torch.float32)
    if dtype == torch.bfloat16:
        scale_quant = scale.npu()
    else:
        scale_quant = torch_npu.npu_trans_quant_param(scale.npu(), None)

    expected_output = op_func(x1.to(torch.int32), x2.to(torch.int32), scale)
    op_func_compiled = get_op_func_compiled()

    custom_output = op_func_compiled(x1_npu, x2_nz, scale_quant, output_dtype=dtype)
    AssertRtolEqual(expected_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.01)
