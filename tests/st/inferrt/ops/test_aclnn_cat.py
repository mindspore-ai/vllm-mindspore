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
"""Tests for torch.cat operation."""
import pytest
import torch

from ms_inferrt.torch import fx_mlir_backend as backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


# pylint: disable=redefined-builtin
def op_func(input, axis):
    """golden"""
    return torch.cat(input, axis)


def get_op_func_compiled():
    """cat op"""
    def custom_op_func(x, axis):
        return torch.cat(x, axis)
    return torch.compile(custom_op_func, backend=backend)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shapes,axis,dtypes", [
    ([2, 5], 0, torch.float16),
    ([3, 4, 5], 1, torch.bfloat16),
    ([1, 10], -1, torch.float32),
    ([5, 3, 2], 2, torch.bfloat16),
    ([2, 3, 4, 5], 3, torch.float32)
])
def test_cat(shapes, axis, dtypes, ):
    """
    Feature: Test aclnn cat
    Description: Test aclnn cat
    Expectation: The result is correct
    """
    cpu_input0 = torch.rand(shapes, dtype=dtypes)
    cpu_input1 = torch.rand(shapes, dtype=dtypes)
    npu_input0 = cpu_input0.npu()
    npu_input1 = cpu_input1.npu()

    cpu_output = op_func((cpu_input0, cpu_input1), axis)
    op_func_compiled = get_op_func_compiled()
    list_in_npu = [npu_input0, npu_input1]
    npu_output = op_func_compiled(list_in_npu, axis)
    npu_output_cpu = npu_output.cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)
