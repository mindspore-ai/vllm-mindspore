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
"""Tests for binary operations (add, sub, mul, div, div_mod)."""
import numpy as np
import pytest
import torch

from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def add_scalar(x):
    return x + 2


def sub_scalar(x):
    return x - 2


def mul_scalar(x):
    return x * 2


def div_scalar(x):
    return x / 2


def div_mod_scalar(x):
    return x // 2


def scalar_forward(op_func, dtype, shape, compiled_func):
    """
    scalar forward function
    Args:
        op_func: The operation function for CPU computation.
        dtype: The data type of the input.
        shape: The shape of the input tensor.
        compiled_func: The compiled op function.
    """
    if np.issubdtype(dtype, np.integer):
        cpu_input = np.random.randint(1, 100, shape).astype(dtype)
        prec = 0
    else:
        if dtype == np.float16:
            prec = 0.001
        else:
            prec = 0.0001
        cpu_input = np.random.uniform(1, 10, shape).astype(dtype)

    npu_input = torch.from_numpy(cpu_input).npu()

    cpu_output = op_func(cpu_input)
    npu_output = compiled_func(npu_input).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output, prec)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(1024, 1024), (256, 512)])
def test_add_tensor_scalar(shape):
    """
    Feature: Test scalar add operation
    Description: Test scalar add with fp32 inputs
    Expectation: The result is correct
    """
    compiled_op = torch.compile(add_scalar, backend=backend, dynamic=True, fullgraph=False)
    scalar_forward(add_scalar, np.float32, shape, compiled_op)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(1024, 1024), (256, 512)])
def test_sub_tensor_scalar(shape):
    """
    Feature: Test scalar sub operation
    Description: Test scalar sub with fp32 inputs
    Expectation: The result is correct
    """
    compiled_op = torch.compile(sub_scalar, backend=backend, dynamic=True, fullgraph=False)
    scalar_forward(sub_scalar, np.float32, shape, compiled_op)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(1024, 1024), (256, 512)])
def test_mul_tensor_scalar(shape):
    """
    Feature: Test scalar mul operation
    Description: Test scalar mul with fp32 inputs
    Expectation: The result is correct
    """
    compiled_op = torch.compile(mul_scalar, backend=backend, dynamic=True, fullgraph=False)
    scalar_forward(mul_scalar, np.float32, shape, compiled_op)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(1024, 1024), (256, 512)])
def test_div_tensor_scalar(shape):
    """
    Feature: Test scalar div operation
    Description: Test scalar div with fp32 inputs
    Expectation: The result is correct
    """
    compiled_op = torch.compile(div_scalar, backend=backend, dynamic=True, fullgraph=False)
    scalar_forward(div_scalar, np.float32, shape, compiled_op)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(1024, 1024), (256, 512)])
def test_div_mod_tensor_scalar(shape):
    """
    Feature: Test scalar div_mod (floor div) operation
    Description: Test scalar div_mod with fp32 inputs
    Expectation: The result is correct
    """
    compiled_op = torch.compile(div_mod_scalar, backend=backend, dynamic=True, fullgraph=False)
    scalar_forward(div_mod_scalar, np.float32, shape, compiled_op)


def add_scalar_tensor(x):
    return 2 + x


def mul_scalar_tensor(x):
    return 2 * x


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(1024, 1024), (256, 512)])
def test_add_scalar_tensor(shape):
    """
    Feature: Test reverse scalar add operation
    Description: Test reverse scalar add (2 + x) with fp32 inputs
    Expectation: The result is correct
    """
    compiled_op = torch.compile(add_scalar_tensor, backend=backend, dynamic=True, fullgraph=False)
    scalar_forward(add_scalar_tensor, np.float32, shape, compiled_op)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(1024, 1024), (256, 512)])
def test_mul_scalar_tensor(shape):
    """
    Feature: Test reverse scalar mul operation
    Description: Test reverse scalar mul (2 * x) with fp32 inputs
    Expectation: The result is correct
    """
    compiled_op = torch.compile(mul_scalar_tensor, backend=backend, dynamic=True, fullgraph=False)
    scalar_forward(mul_scalar_tensor, np.float32, shape, compiled_op)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("x", [5, 10, 15])
def test_add_scalar_scalar(x):
    compiled_op = torch.compile(add_scalar, backend=backend, dynamic=True, fullgraph=False)
    out = compiled_op(x)
    expected = x + 2
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_sub_scalar_scalar():
    compiled_op = torch.compile(sub_scalar, backend=backend, dynamic=True, fullgraph=False)
    x = 5
    out = compiled_op(x)
    expected = x - 2
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_mul_scalar_scalar():
    compiled_op = torch.compile(mul_scalar, backend=backend, dynamic=True, fullgraph=False)
    x = 5
    out = compiled_op(x)
    expected = x * 2
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_div_scalar_scalar():
    compiled_op = torch.compile(div_scalar, backend=backend, dynamic=True, fullgraph=False)
    x = 5
    out = compiled_op(x)
    expected = x / 2
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_div_mod_scalar_scalar():
    compiled_op = torch.compile(div_mod_scalar, backend=backend, dynamic=True, fullgraph=False)
    x = 5
    out = compiled_op(x)
    expected = x // 2
    AssertRtolEqual(out, expected)
