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
"""
Tests for op capability check and fallback to custom_call.

When an op selected by _get_op is not registered on the target backend (e.g. CPU),
fx_backend calls check_op_support and falls back to Op.custom_call so the graph
can still run via Python/custom path. These tests verify the flow on CPU using
add operation (which is registered on CPU, exercising the supported path).
"""
# pylint: disable=wrong-import-position, ungrouped-imports
import os
os.environ["USE_NPU"] = "0"
os.environ["USE_ASCEND"] = "0"
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["TASK_QUEUE_ENABLE"] = "0"

import numpy as np
import pytest
import torch

from ms_inferrt.torch.fx_backend import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def add_func(x1, x2):
    """Add operation function."""
    return torch.add(x1, x2)


def add_forward(dtype, shape, compiled_func):
    """
    Add forward function.
    Args:
        dtype: The data type of the input.
        shape: The shape of the input tensor.
        compiled_func: The compiled op function.
    """
    if np.issubdtype(dtype, np.integer):
        cpu_input0 = np.random.randint(-100, 100, shape).astype(dtype)
        cpu_input1 = np.random.randint(-100, 100, shape).astype(dtype)
        prec = 0
    else:
        if dtype == np.float16:
            prec = 0.001
        else:
            prec = 0.0001
        cpu_input0 = np.random.uniform(-1, 1, shape).astype(dtype)
        cpu_input1 = np.random.uniform(-1, 1, shape).astype(dtype)

    cpu_input0_t = torch.from_numpy(cpu_input0).to(device="cpu")
    cpu_input1_t = torch.from_numpy(cpu_input1).to(device="cpu")

    expected = add_func(cpu_input0_t, cpu_input1_t).numpy()
    result = compiled_func(cpu_input0_t, cpu_input1_t).detach().numpy()

    AssertRtolEqual(expected, result, prec)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(1,), (64, 10), (32, 3, 3), (256, 512)])
def test_op_fallback_add_fp32(shape):
    """
    Feature: Op capability check and fallback to custom_call
    Description: Add on CPU (registered) - verify fx_backend flow with fp32
    Expectation: Result correct when op is supported on target device
    """
    compiled_op = torch.compile(add_func, backend=fx_backend)
    add_forward(np.float32, shape, compiled_op)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(1,), (64, 10), (32, 3, 3), (256, 512)])
def test_op_fallback_add_int32(shape):
    """
    Feature: Op capability check and fallback to custom_call
    Description: Add on CPU (registered) - verify fx_backend flow with int32
    Expectation: Result correct when op is supported on target device
    """
    compiled_op = torch.compile(add_func, backend=fx_backend)
    add_forward(np.int32, shape, compiled_op)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(1,), (64, 10), (32, 3, 3)])
def test_op_fallback_add_fp16(shape):
    """
    Feature: Op capability check and fallback to custom_call
    Description: Add on CPU (registered) - verify fx_backend flow with fp16
    Expectation: Result correct when op is supported on target device
    """
    compiled_op = torch.compile(add_func, backend=fx_backend)
    add_forward(np.float16, shape, compiled_op)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_op_fallback_add_with_alpha():
    """
    Feature: Op capability check and fallback to custom_call
    Description: Add with alpha on CPU - verify capability check path
    Expectation: Result correct when op is supported on target device
    """

    def add_with_alpha(x1, x2):
        return torch.add(x1, x2, alpha=2.0)

    compiled_op = torch.compile(add_with_alpha, backend=fx_backend)

    cpu_input0 = torch.randn(4, 4, dtype=torch.float32)
    cpu_input1 = torch.randn(4, 4, dtype=torch.float32)

    expected = add_with_alpha(cpu_input0, cpu_input1)
    result = compiled_op(cpu_input0, cpu_input1)

    AssertRtolEqual(expected.numpy(), result.detach().numpy())


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.skip(reason="Conflicts with add_scalar registered on CPU that returns scalar type. "
                        "Will be enabled after type matching is implemented.")
def test_op_fallback_add_scalar():
    """
    Feature: Op capability check and fallback to custom_call
    Description: Add with scalar on CPU - verify capability check path
    Expectation: Result correct when op is supported on target device
    """

    def add_scalar(x):
        return x + 1.0

    compiled_op = torch.compile(add_scalar, backend=fx_backend)

    cpu_input = torch.randn(2, 2, dtype=torch.float32)
    expected = add_scalar(cpu_input)
    result = compiled_op(cpu_input)

    AssertRtolEqual(expected.numpy(), result.detach().numpy())
