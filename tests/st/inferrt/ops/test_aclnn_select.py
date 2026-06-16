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
"""Tests for aten.select.int via fx_backend (select_view)."""
import numpy as np
import pytest
import torch

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def select_aten_int(x, dim, index):
    """torch.ops.aten.select.int style"""
    return torch.ops.aten.select.int(x, dim, index)


def select_torch_select(x, dim, index):
    """torch.select style"""
    return torch.select(x, dim, index)


def select_string(x, dim, index):
    """string 'select' via compile"""
    return x.select(dim, index)


def select_forward(dtype, shape, dim, index, op_func, compiled_func):
    """Forward execution helper for select tests."""
    if np.issubdtype(dtype, np.integer):
        cpu_input = np.random.randint(-100, 100, shape).astype(dtype)
    else:
        cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)

    npu_input = torch.from_numpy(cpu_input).npu()

    ori_output = op_func(npu_input, dim, index).detach().cpu().numpy()
    npu_output = compiled_func(npu_input, dim, index).detach().cpu().numpy()

    AssertRtolEqual(ori_output, npu_output)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(128, 256, 64)])
@pytest.mark.parametrize("op_func", [
    select_aten_int, select_torch_select, select_string
])
@pytest.mark.parametrize("dim,index", [
    (0, 5),
    (1, 10),
    (2, 20),
    (-1, 30),
])
# pylint: disable=redefined-outer-name
def test_select_int_fp32(shape, op_func, dim, index):
    """
    Feature: Test aten.select.int via fx_backend
    Description: Test aten.select.int with fp32 3D inputs, covering call styles and dim
    Expectation: The result is correct
    """
    compiled_op = torch.compile(op_func, backend=backend)
    select_forward(np.float32, shape, dim, index, op_func, compiled_op)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(32, 64)])
@pytest.mark.parametrize("op_func", [select_aten_int])
@pytest.mark.parametrize("dim,index", [(0, 3)])
def test_select_int_fp16(shape, op_func, dim, index):
    """
    Feature: Test aten.select.int fp16
    Description: Test with float16
    Expectation: The result is correct
    """
    compiled_op = torch.compile(op_func, backend=backend)
    select_forward(np.float16, shape, dim, index, op_func, compiled_op)
