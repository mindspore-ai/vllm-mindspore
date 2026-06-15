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
"""Tests for neg operation."""
import pytest
import torch

from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(x):
    """golden for neg"""
    return torch.neg(x)


def aten_neg_default(x):
    """golden for aten.neg.default"""
    return torch.ops.aten.neg.default(x)


def get_op_func_compiled():
    """neg op compiled"""
    def custom_op_func(x):
        return torch.neg(x)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8, 16), (64, 32), (128, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_neg(shape, dtype):
    """
    Feature: Test op neg
    Description: Test op neg with multi-dimensional tensor inputs
    Expectation: The result is correct
    """
    cpu_input = torch.rand(shape, dtype=dtype)
    npu_input = cpu_input.npu()
    cpu_output = op_func(cpu_input)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(npu_input)
    npu_output_cpu = npu_output.cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_aten_neg_default(dtype):
    """
    Feature: Test aten.neg.default
    Description: Verify exact aten neg overload through fx_backend.
    Expectation: The result is correct
    """
    cpu_input = torch.rand((8, 16), dtype=dtype)
    npu_input = cpu_input.npu()
    cpu_output = aten_neg_default(cpu_input)
    op_func_compiled = torch.compile(aten_neg_default, backend=backend)
    npu_output = op_func_compiled(npu_input)
    AssertRtolEqual(cpu_output, npu_output.cpu())
