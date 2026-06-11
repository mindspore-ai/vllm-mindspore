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
"""Tests for aclnn div operation."""
import numpy as np
import pytest
import torch

from ms_inferrt.torch import fx_mlir_backend as backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual

def div_op(x, y):
    return x / y

def get_op_func_compiled():
    return torch.compile(div_op, backend=backend)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", ([2, 3], [512, 256], [1024, 512]))
def test_div(shape):
    """
    Feature: Test aclnn div
    Description: Test aclnn div and with different dtype inputs and different shape
    Expectation: The result is correct
    """

    tensor_x_cpu = np.random.uniform(-1, 1, shape).astype(np.float16)
    tensor_y_cpu = np.random.uniform(-1, 1, shape).astype(np.float16)

    tensor_x = torch.from_numpy(tensor_x_cpu).npu()
    tensor_y = torch.from_numpy(tensor_y_cpu).npu()

    result_operate = div_op(tensor_x_cpu, tensor_y_cpu)

    compile_op = get_op_func_compiled()
    result_compile_op = compile_op(tensor_x, tensor_y).detach().cpu().numpy()

    AssertRtolEqual(result_operate, result_compile_op)

def aten_div_dynamic_op(x, y):
    b = x.size(0)
    return torch.ops.aten.div.Tensor(x[:b], y[:b])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8,), (4, 8), (16, 32), (64, 128), (2, 4, 8), (1, 8, 16, 32)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_aten_div_dynamic(shape, dtype):
    """
Feature: Test aten div with dynamic shapes.
    Description: Test aten.div.Tensor with various shapes.
    Expectation: The result matches eager mode.
    """
    compiled_op = torch.compile(aten_div_dynamic_op, backend=backend)
    prec = 0.001 if dtype == np.float16 else 0.0001
    cpu_input0 = np.random.uniform(0.5, 2, shape).astype(dtype)
    cpu_input1 = np.random.uniform(0.5, 2, shape).astype(dtype)
    npu_input0 = torch.from_numpy(cpu_input0).npu()
    npu_input1 = torch.from_numpy(cpu_input1).npu()
    cpu_output = aten_div_dynamic_op(torch.from_numpy(cpu_input0), torch.from_numpy(cpu_input1)).detach().numpy()
    npu_output = compiled_op(npu_input0, npu_input1).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output, prec)

def div_dynamic_op(x, y):
    b = x.size(0)
    return x[:b] / y[:b]


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8,), (4, 8), (16, 32), (64, 128), (2, 4, 8), (1, 8, 16, 32)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_div_dynamic(shape, dtype):
    """
Feature: Test div with dynamic shapes.
    Description: Test torch.div with dynamic input slicing.
    Expectation: The result matches eager mode.
    """
    compiled_op = torch.compile(div_dynamic_op, backend=backend)
    prec = 0.001 if dtype == np.float16 else 0.0001
    cpu_input0 = np.random.uniform(0.5, 2, shape).astype(dtype)
    cpu_input1 = np.random.uniform(0.5, 2, shape).astype(dtype)
    npu_input0 = torch.from_numpy(cpu_input0).npu()
    npu_input1 = torch.from_numpy(cpu_input1).npu()
    cpu_output = div_dynamic_op(torch.from_numpy(cpu_input0), torch.from_numpy(cpu_input1)).detach().numpy()
    npu_output = compiled_op(npu_input0, npu_input1).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output, prec)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8,), (4, 8), (16, 32), (64, 128)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_div_static(shape, dtype):
    """
Feature: Test div with static shapes.
    Description: Test torch.div with fixed shapes.
    Expectation: The result matches eager mode.
    """
    def div_static_op(x, y):
        return torch.div(x, y)

    compiled_op = torch.compile(div_static_op, backend=backend)
    prec = 0.001 if dtype == np.float16 else 0.0001
    cpu_input0 = np.random.uniform(0.5, 2, shape).astype(dtype)
    cpu_input1 = np.random.uniform(0.5, 2, shape).astype(dtype)
    npu_input0 = torch.from_numpy(cpu_input0).npu()
    npu_input1 = torch.from_numpy(cpu_input1).npu()
    cpu_output = div_static_op(torch.from_numpy(cpu_input0), torch.from_numpy(cpu_input1)).detach().numpy()
    npu_output = compiled_op(npu_input0, npu_input1).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output, prec)
