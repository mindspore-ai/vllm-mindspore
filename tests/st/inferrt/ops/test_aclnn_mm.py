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
"""Tests for mm operation."""
import pytest
import torch

from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(x, weight):
    """golden for mm"""
    return torch.mm(x, weight)


def aten_mm_default(x, weight):
    """golden for aten.mm.default"""
    return torch.ops.aten.mm.default(x, weight)


def get_op_func_compiled():
    """mm op compiled"""
    def custom_op_func(x, w):
        return torch.mm(x, w)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("m,n,k", [(16, 32, 64), (32, 64, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm(m, n, k, dtype):
    """
    Feature: Test op mm
    Description: Test op mm with matrix multiplication
    Expectation: The result is correct
    """
    cpu_input = torch.rand(m, k, dtype=dtype)
    cpu_weight = torch.rand(k, n, dtype=dtype)
    npu_input = cpu_input.npu()
    npu_weight = cpu_weight.npu()
    cpu_output = op_func(cpu_input, cpu_weight)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(npu_input, npu_weight)
    npu_output_cpu = npu_output.cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("m,n,k", [(8, 16, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_aten_mm_default(m, n, k, dtype):
    """
    Feature: Test aten.mm.default
    Description: Verify exact aten mm overload through fx_backend.
    Expectation: The result is correct
    """
    cpu_input = torch.rand(m, k, dtype=dtype)
    cpu_weight = torch.rand(k, n, dtype=dtype)
    npu_input = cpu_input.npu()
    npu_weight = cpu_weight.npu()
    cpu_output = aten_mm_default(cpu_input, cpu_weight)
    op_func_compiled = torch.compile(aten_mm_default, backend=backend)
    npu_output = op_func_compiled(npu_input, npu_weight)
    AssertRtolEqual(cpu_output, npu_output.cpu())
