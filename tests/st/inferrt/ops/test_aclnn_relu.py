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
"""Tests for aclnn relu operation."""
import pytest
import torch

from ms_inferrt.torch import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def relu_torch(x):
    """Using torch.relu"""
    return torch.relu(x)


def relu_nn(x):
    """Using torch.nn.functional.relu"""
    return torch.nn.functional.relu(x)


def relu_method(x):
    """Using .relu() tensor method"""
    return x.relu()


def get_op_func_compiled(op_func):
    return torch.compile(op_func, backend=fx_backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", ([2, 3], [15, 64], [1024, 512]))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
@pytest.mark.parametrize("op_func", [relu_torch, relu_nn, relu_method])
def test_relu(shape, dtype, op_func):
    """
    Feature: Test aclnn relu
    Description: Test aclnn relu with different dtypes, shapes and invocation styles
                 (torch.relu, nn.functional.relu, .relu method)
    Expectation: The result is correct
    """
    tensor_x = torch.randn(shape, dtype=dtype, device="npu")

    tensor_x_cpu = tensor_x.cpu()

    result_eager = op_func(tensor_x_cpu)

    compile_op = get_op_func_compiled(op_func)
    result_compile = compile_op(tensor_x).cpu()

    AssertRtolEqual(result_eager, result_compile)
