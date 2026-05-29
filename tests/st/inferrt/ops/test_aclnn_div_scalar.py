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
"""Tests for div_scalar operation."""
import pytest
import torch

from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(x):
    """golden for div_scalar"""
    return x / 2.5


def get_op_func_compiled():
    """div_scalar op compiled"""
    def custom_op_func(x):
        return x / 2.5
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8, 16), (64, 32), (128, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_div_scalar(shape, dtype):
    """
    Feature: Test op div_scalar
    Description: Test op div_scalar with multi-dimensional tensor inputs
    Expectation: The result is correct
    """
    cpu_input = torch.rand(shape, dtype=dtype)
    npu_input = cpu_input.npu()
    cpu_output = op_func(cpu_input)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(npu_input)
    npu_output_cpu = npu_output.cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)
