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
"""Tests for torch.permute operation."""
import pytest
import torch

from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


_permute_dims_4d = [
    (0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1),
    (0, 3, 1, 2), (0, 3, 2, 1), (1, 0, 2, 3), (1, 0, 3, 2),
    (1, 2, 0, 3), (1, 2, 3, 0), (1, 3, 0, 2), (1, 3, 2, 0),
    (2, 0, 1, 3), (2, 0, 3, 1), (2, 1, 0, 3), (2, 1, 3, 0),
    (2, 3, 0, 1), (2, 3, 1, 0), (3, 0, 1, 2), (3, 0, 2, 1),
    (3, 1, 0, 2), (3, 1, 2, 0), (3, 2, 0, 1), (3, 2, 1, 0),
]

_permute_dims_3d = [
    (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0),
]

_permute_dims_2d = [
    (0, 1), (1, 0),
]


# pylint: disable=redefined-builtin
def op_func(input, dims):
    """golden for permute"""
    return input.permute(dims)


def get_op_func_compiled():
    """permute op compiled"""
    def custom_op_func(x, dims):
        return torch.permute(x, dims)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8, 8, 512, 128), (2, 4, 256, 64)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dims", _permute_dims_4d)
def test_permute_4d(shape, dtype, dims):
    """
    Feature: Test aclnn permute
    Description: Test aclnn permute with 4D tensor inputs
    Expectation: The result is correct
    """
    if dtype in [torch.int32, torch.int64]:
        cpu_input = torch.randint(0, 100, shape, dtype=dtype)
    else:
        cpu_input = torch.rand(shape, dtype=dtype)
    npu_input = cpu_input.npu()
    cpu_output = op_func(cpu_input, dims)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(npu_input, dims)
    npu_output_cpu = npu_output.cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(4, 8, 16), (2, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dims", _permute_dims_3d)
def test_permute_3d(shape, dtype, dims):
    """
    Feature: Test aclnn permute
    Description: Test aclnn permute with 3D tensor inputs
    Expectation: The result is correct
    """
    cpu_input = torch.rand(shape, dtype=dtype)
    npu_input = cpu_input.npu()
    cpu_output = op_func(cpu_input, dims)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(npu_input, dims)
    npu_output_cpu = npu_output.cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8, 16), (4, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dims", _permute_dims_2d)
def test_permute_2d(shape, dtype, dims):
    """
    Feature: Test aclnn permute
    Description: Test aclnn permute with 2D tensor inputs
    Expectation: The result is correct
    """
    cpu_input = torch.rand(shape, dtype=dtype)
    npu_input = cpu_input.npu()
    cpu_output = op_func(cpu_input, dims)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(npu_input, dims)
    npu_output_cpu = npu_output.cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)
