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
"""Tests for alias operation."""
import pytest
import torch

from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(x):
    """golden for alias"""
    return torch.ops.aten.alias.default(x)


def get_op_func_compiled():
    """alias op compiled"""
    def custom_op_func(x):
        return torch.ops.aten.alias.default(x)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8, 16)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int32])
def test_alias(shape, dtype):
    """
    Feature: Test op alias
    Description: Test op alias with multi-dimensional tensor inputs
    Expectation: The result is correct
    """
    if dtype in [torch.float16, torch.float32]:
        cpu_input = torch.randn(shape, dtype=dtype)
    else:
        cpu_input = torch.randint(0, 100, shape, dtype=dtype)
    cpu_output = op_func(cpu_input)
    op_func_compiled = get_op_func_compiled()
    cpu_output_inferrt = op_func_compiled(cpu_input)
    AssertRtolEqual(cpu_output, cpu_output_inferrt)
