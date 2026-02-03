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
"""Tests for torch.clone operation."""
import pytest
import torch

from mrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


# pylint: disable=redefined-builtin
def op_func(input, memory_formats):
    """golden"""
    return torch.clone(input, memory_format=memory_formats)


# pylint: disable=redefined-builtin
def op_func_default(input):
    """golden"""
    return torch.clone(input)


def get_op_func_compiled():
    """clone op"""
    def custom_op_func(x, memory_formats):
        return torch.clone(x, memory_format=memory_formats)
    return torch.compile(custom_op_func, backend=backend)


def get_op_func_compiled_default():
    """clone op"""
    def custom_op_func(x):
        return torch.clone(x)
    return torch.compile(custom_op_func, backend=backend)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shapes", [[9, 2], [32, 16, 4096], [2, 5, 9, 2]])
@pytest.mark.parametrize("dtypes", [torch.float16, torch.bfloat16, torch.float32])
def test_clone_default(shapes, dtypes, ):
    """
    Feature: Test aclnn clone default
    Description: Test aclnn clone
    Expectation: The result is correct
    """
    cpu_input0 = torch.rand(shapes, dtype=dtypes)
    npu_input0 = cpu_input0.npu()

    cpu_output = op_func_default(cpu_input0)
    op_func_compiled = get_op_func_compiled_default()
    npu_output = op_func_compiled(npu_input0)
    npu_output_cpu = npu_output.cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shapes", [[9, 2], [32, 16, 4096], [2, 5, 9, 2]])
@pytest.mark.parametrize("memory_formats", [torch.contiguous_format])
@pytest.mark.parametrize("dtypes", [torch.float16, torch.bfloat16, torch.float32])
def test_clone_contiguous_format(shapes, memory_formats, dtypes, ):
    """
    Feature: Test aclnn clone contiguous_format
    Description: Test aclnn clone
    Expectation: The result is correct
    """
    cpu_input0 = torch.rand(shapes, dtype=dtypes)
    npu_input0 = cpu_input0.npu()

    cpu_output = op_func(cpu_input0, memory_formats)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(npu_input0, memory_formats)
    npu_output_cpu = npu_output.cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shapes", [[9, 2], [32, 16, 4096], [2, 5, 9, 2]])
@pytest.mark.parametrize("memory_formats", [torch.preserve_format])
@pytest.mark.parametrize("dtypes", [torch.float16, torch.bfloat16, torch.float32])
def test_clone_preserve_format(shapes, memory_formats, dtypes, ):
    """
    Feature: Test aclnn clone preserve_format
    Description: Test aclnn clone
    Expectation: The result is correct
    """
    cpu_input0 = torch.rand(shapes, dtype=dtypes)
    npu_input0 = cpu_input0.npu()

    cpu_output = op_func(cpu_input0, memory_formats)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(npu_input0, memory_formats)
    npu_output_cpu = npu_output.cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shapes", [[2, 5, 9, 2], [32, 16, 40, 96]])
@pytest.mark.parametrize("memory_formats", [torch.channels_last])
@pytest.mark.parametrize("dtypes", [torch.float16, torch.bfloat16, torch.float32])
def test_clone_channels_last(shapes, memory_formats, dtypes, ):
    """
    Feature: Test aclnn clone channels_last
    Description: Test aclnn clone
    Expectation: The result is correct
    """
    cpu_input0 = torch.rand(shapes, dtype=dtypes)
    npu_input0 = cpu_input0.npu()

    cpu_output = op_func(cpu_input0, memory_formats)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(npu_input0, memory_formats)
    npu_output_cpu = npu_output.cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shapes", [[2, 5, 9, 2, 1], [32, 16, 40, 96, 1]])
@pytest.mark.parametrize("memory_formats", [torch.channels_last_3d])
@pytest.mark.parametrize("dtypes", [torch.float16, torch.bfloat16, torch.float32])
def test_clone_channels_last_3d(shapes, memory_formats, dtypes, ):
    """
    Feature: Test aclnn clone channels_last_3d
    Description: Test aclnn clone
    Expectation: The result is correct
    """
    cpu_input0 = torch.rand(shapes, dtype=dtypes)
    npu_input0 = cpu_input0.npu()

    cpu_output = op_func(cpu_input0, memory_formats)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(npu_input0, memory_formats)
    npu_output_cpu = npu_output.cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)
