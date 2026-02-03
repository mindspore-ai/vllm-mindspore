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
"""Tests for aclnn mul operation."""
import numpy as np
import torch
from torch_npu.testing.common_utils import create_common_tensor

from mrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(input1, input2):
    """op function for mul"""
    return torch.mul(input1, input2)

def mul_forward(shape_format, op_func_compiled):
    """
    mul forward function
    Args:
        shape_format: list of [dtype, format, shape]
        op_func_compiled: The compiled op function.
    """
    for item in shape_format:
        if 0 in item[2]:
            # skip the shape that contains 0
            continue
        cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
        cpu_input2, npu_input2 = create_common_tensor(item, 1, 100)
        if cpu_input1.dtype == torch.float16:
            cpu_input1 = cpu_input1.to(torch.float32)
        if cpu_input2.dtype == torch.float16:
            cpu_input2 = cpu_input2.to(torch.float32)
        cpu_output = op_func(cpu_input1, cpu_input2).detach().numpy()
        npu_output = op_func_compiled(npu_input1, npu_input2).detach().cpu().numpy()
        cpu_output = cpu_output.astype(npu_output.dtype)
        AssertRtolEqual(cpu_output, npu_output)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_muls_shape_format_fp16():
    """
    Feature: Test aclnn mul
    Description: Test aclnn mul with fp16 inputs
    Expectation: The result is correct
    """
    format_list = [2]
    shape_list = [(1,), (64, 10), (32, 3, 3), (256, 2048, 7, 7), (2, 0, 2)]
    shape_format = [
        [np.float16, i, j] for i in format_list for j in shape_list
    ]
    op_func_compiled = torch.compile(op_func, backend=backend)
    mul_forward(shape_format, op_func_compiled)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_muls_shape_format_fp32():
    """
    Feature: Test aclnn mul
    Description: Test aclnn mul with fp32 inputs
    Expectation: The result is correct
    """
    format_list = [2]
    shape_list = [(1,), (64, 10), (32, 3, 3), (256, 2048, 7, 7)]
    shape_format = [
        [np.float32, i, j] for i in format_list for j in shape_list
    ]
    op_func_compiled = torch.compile(op_func, backend=backend)
    mul_forward(shape_format, op_func_compiled)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_muls_shape_format_complex():
    """
    Feature: Test aclnn mul
    Description: Test aclnn mul with complex inputs
    Expectation: The result is correct
    """

    op_func_compiled = torch.compile(op_func, backend=backend)

    format_list = [2]
    shape_list = [(1,), (64, 10), (32, 3, 3), (256, 2048, 7, 7)]
    shape_format = [
        [torch.cfloat, i, j] for i in format_list for j in shape_list
    ]
    for item in shape_format:
        cpu_input1 = torch.randn(item[1], dtype=item[0])
        npu_input1 = cpu_input1.npu()
        cpu_input2 = torch.randn(item[1], dtype=item[0])
        npu_input2 = cpu_input2.npu()
        cpu_output = op_func(cpu_input1, cpu_input2).detach().numpy()
        npu_output = op_func_compiled(npu_input1, npu_input2).detach().cpu().numpy()
        if item[0] == torch.cfloat:
            cpu_output = cpu_output.astype(np.float32)
            npu_output = npu_output.astype(np.float32)
        AssertRtolEqual(cpu_output, npu_output)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_muls_shape_format_bool():
    """
    Feature: Test aclnn mul
    Description: Test aclnn mul with bool inputs
    Expectation: The result is correct
    """

    op_func_compiled = torch.compile(op_func, backend=backend)

    format_list = [0]
    shape_list = [(1,), (64, 10), (32, 3, 3), (256, 2048, 7, 7)]
    for format_idx in format_list:
        for shape in shape_list:
            item = [np.int32, format_idx, shape]
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 1, 100)

            cpu_bool1 = cpu_input1 > 50
            npu_bool1 = npu_input1 > 50
            cpu_bool2 = cpu_input2 > 50
            npu_bool2 = npu_input2 > 50

            cpu_output = op_func(cpu_bool1, cpu_bool2).detach().numpy()
            npu_output = op_func_compiled(npu_bool1, npu_bool2).detach().cpu().numpy()

            AssertRtolEqual(cpu_output, npu_output)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_muls_mix_dtype():
    """
    Feature: Test aclnn mul
    Description: Test aclnn mul with mixed dtypes (int32 and float32)
    Expectation: The result is correct
    """

    op_func_compiled = torch.compile(op_func, backend=backend)

    cpu_input1, npu_input1 = create_common_tensor([np.int32, 0, (2, 3)], 1, 100)
    cpu_input2, npu_input2 = create_common_tensor([np.float32, 0, (2, 3)], 1, 100)

    cpu_output = op_func(cpu_input1, cpu_input2).detach().numpy()
    npu_output = op_func_compiled(npu_input1, npu_input2).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output)
