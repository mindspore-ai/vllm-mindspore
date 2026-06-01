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
"""Tests for aclnn leaky_relu operation."""
import pytest
import torch

from ms_inferrt.torch import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def leaky_relu_default(x):
    """Using torch.nn.functional.leaky_relu with default negative_slope=0.01"""
    return torch.nn.functional.leaky_relu(x)


def leaky_relu_custom(x):
    """Using torch.nn.functional.leaky_relu with custom negative_slope"""
    return torch.nn.functional.leaky_relu(x, negative_slope=0.1)


def get_op_func_compiled(op_func):
    return torch.compile(op_func, backend=fx_backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", ([2, 3], [15, 64], [1024, 512]))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
@pytest.mark.parametrize("op_func", [leaky_relu_default, leaky_relu_custom])
def test_leaky_relu(shape, dtype, op_func):
    """
    Feature: Test aclnn leaky_relu
    Description: Test aclnn leaky_relu with different dtypes, shapes and negative_slope
    Expectation: The result is correct
    """
    tensor_x = torch.randn(shape, dtype=dtype, device="npu")

    tensor_x_cpu = tensor_x.cpu()

    result_eager = op_func(tensor_x_cpu)

    compile_op = get_op_func_compiled(op_func)
    result_compile = compile_op(tensor_x).cpu()

    AssertRtolEqual(result_eager, result_compile)
