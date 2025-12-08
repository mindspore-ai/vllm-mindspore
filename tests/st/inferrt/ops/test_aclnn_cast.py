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

import pytest
import numpy as np
import torch

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual
from mrt.torch import fx_mlir_backend as backend


def op_func(x, dtype):
    """op function for cast"""
    return x.to(dtype)


def get_op_func_compiled():
    def custom_op_func(x, dtype):
        return x.to(dtype)
    return torch.compile(custom_op_func, backend=backend)

def long_op_func(input_tensor):
    return input_tensor.long()

def get_long_op_func_compiled():
    def custom_op_func(input_tensor):
        return input_tensor.long()
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
@pytest.mark.parametrize("input_dtype,output_dtype", [
    (torch.float32, torch.float16),
    (torch.float16, torch.float32),
    (torch.float32, torch.int64),
    (torch.int64, torch.float32),
    (torch.int32, torch.float32),
    (torch.float32, torch.int32),
])
def test_cast_basic(pipeline, monkeypatch, input_dtype, output_dtype):
    """
    Feature: Test aclnn cast
    Description: Test aclnn cast with basic dtype conversions
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    shape = (2, 3, 4)
    if input_dtype in (torch.float32, torch.float16):
        cpu_input = torch.randn(shape, dtype=input_dtype)
    else:
        cpu_input = torch.randint(-100, 100, shape, dtype=input_dtype)
    npu_input = cpu_input.npu()

    cpu_output = op_func(cpu_input, output_dtype)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(npu_input, output_dtype).detach().cpu()

    AssertRtolEqual(cpu_output.numpy(), npu_output.numpy())


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
@pytest.mark.parametrize("shape", [
    (1,),
    (64, 10),
    (32, 3, 3),
    (256, 2048, 7, 7),
])
def test_cast_shapes(pipeline, monkeypatch, shape):
    """
    Feature: Test aclnn cast
    Description: Test aclnn cast with different shapes
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    cpu_input = torch.randn(shape, dtype=torch.float32)
    npu_input = cpu_input.npu()

    cpu_output = op_func(cpu_input, torch.float16)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(npu_input, torch.float16).detach().cpu()

    AssertRtolEqual(cpu_output.numpy(), npu_output.numpy())

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
@pytest.mark.parametrize("shape", [
    (1,),
    (64, 10),
    (32, 3, 3),
    (256, 2048, 7, 7),
])
def test_cast_long_shapes(pipeline, monkeypatch, shape):
    """
    Feature: Test aclnn cast
    Description: Test aclnn cast with different shapes
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    cpu_input = torch.randn(shape, dtype=torch.float32)
    npu_input = cpu_input.npu()

    cpu_output = long_op_func(cpu_input)
    op_func_compiled = get_long_op_func_compiled()
    npu_output = op_func_compiled(npu_input).detach().cpu()
    AssertRtolEqual(cpu_output, npu_output)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_cast_float_to_int(pipeline, monkeypatch):
    """
    Feature: Test aclnn cast
    Description: Test aclnn cast from float32 to int64
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    cpu_input = torch.randn(2, 5, dtype=torch.float32) * 10.0
    npu_input = cpu_input.npu()

    cpu_output = op_func(cpu_input, torch.int64)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(npu_input, torch.int64).detach().cpu()

    AssertRtolEqual(cpu_output.numpy(), npu_output.numpy())


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_cast_int_to_float(pipeline, monkeypatch):
    """
    Feature: Test aclnn cast
    Description: Test aclnn cast from int64 to float32
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    cpu_input = torch.randint(-100, 100, (2, 5), dtype=torch.int64)
    npu_input = cpu_input.npu()

    cpu_output = op_func(cpu_input, torch.float32)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(npu_input, torch.float32).detach().cpu()

    AssertRtolEqual(cpu_output.numpy(), npu_output.numpy())
