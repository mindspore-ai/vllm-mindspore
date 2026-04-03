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
"""Tests for binary operations with scalar inputs on CPU."""
# pylint: disable=wrong-import-position, ungrouped-imports
import os
os.environ["USE_NPU"] = "0"
os.environ["USE_ASCEND"] = "0"
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["TASK_QUEUE_ENABLE"] = "0"

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


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("x", [5, 10, 15])
def test_add_scalar_scalar(x):
    """
    Feature: Test scalar add operation with scalar inputs
    Description: Test add operation with scalar inputs (scalar + scalar)
    Expectation: The result is correct
    """
    compiled_op = torch.compile(add_scalar, backend=backend, dynamic=True, fullgraph=False)
    out = compiled_op(x)
    expected = x + 2
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_sub_scalar_scalar():
    """
    Feature: Test scalar sub operation with scalar inputs
    Description: Test sub operation with scalar inputs (scalar - scalar)
    Expectation: The result is correct
    """
    compiled_op = torch.compile(sub_scalar, backend=backend, dynamic=True, fullgraph=False)
    x = 5
    out = compiled_op(x)
    expected = x - 2
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_mul_scalar_scalar():
    """
    Feature: Test scalar mul operation with scalar inputs
    Description: Test mul operation with scalar inputs (scalar * scalar)
    Expectation: The result is correct
    """
    compiled_op = torch.compile(mul_scalar, backend=backend, dynamic=True, fullgraph=False)
    x = 5
    out = compiled_op(x)
    expected = x * 2
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_div_scalar_scalar():
    """
    Feature: Test scalar div operation with scalar inputs
    Description: Test div operation with scalar inputs (scalar / scalar)
    Expectation: The result is correct
    """
    compiled_op = torch.compile(div_scalar, backend=backend, dynamic=True, fullgraph=False)
    x = 5
    out = compiled_op(x)
    expected = x / 2
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_div_mod_scalar_scalar():
    """
    Feature: Test scalar div_mod operation with scalar inputs
    Description: Test div_mod operation with scalar inputs (scalar // scalar)
    Expectation: The result is correct
    """
    compiled_op = torch.compile(div_mod_scalar, backend=backend, dynamic=True, fullgraph=False)
    x = 5
    out = compiled_op(x)
    expected = x // 2
    AssertRtolEqual(out, expected)
