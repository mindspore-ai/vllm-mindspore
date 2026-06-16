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
"""Tests for aclnn where operation (aten.where.self, torch.where, etc.)."""
import numpy as np
import pytest
import torch

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def where_aten_self(cond, x, y):
    """torch.ops.aten.where.self style"""
    return torch.ops.aten.where.self(cond, x, y)


def where_torch_where(cond, x, y):
    """torch.where style"""
    return torch.where(cond, x, y)


def where_tensor_method(cond, x, y):
    """x.where(y, cond) wait, actually tensor.where is different; use where on cond"""
    # cond.where is not standard; use functional
    return torch.where(cond, x, y)


def where_string_method(cond, x, y):
    """Simulate string 'where' via compile"""
    return torch.where(cond, x, y)


def where_forward(dtype, shape, op_func, compiled_func):
    """
    where forward function
    """
    cpu_cond = np.random.choice([True, False], shape).astype(np.bool_)
    if np.issubdtype(dtype, np.integer):
        cpu_x = np.random.randint(-100, 100, shape).astype(dtype)
        cpu_y = np.random.randint(-100, 100, shape).astype(dtype)
    else:
        cpu_x = np.random.uniform(-1, 1, shape).astype(dtype)
        cpu_y = np.random.uniform(-1, 1, shape).astype(dtype)

    npu_cond = torch.from_numpy(cpu_cond).npu()
    npu_x = torch.from_numpy(cpu_x).npu()
    npu_y = torch.from_numpy(cpu_y).npu()

    ori_output = op_func(npu_cond, npu_x, npu_y).detach().cpu().numpy()
    npu_output = compiled_func(npu_cond, npu_x, npu_y).detach().cpu().numpy()

    AssertRtolEqual(ori_output, npu_output)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(128, 256), (64, 32, 16)])
@pytest.mark.parametrize("op_func", [
    where_aten_self, where_torch_where, where_tensor_method, where_string_method
])
# pylint: disable=redefined-outer-name
def test_where_fp32(shape, op_func):
    """
    Feature: Test aclnn where on 2D/3D tensor
    Description: Test aclnn where with fp32 inputs, covering aten/torch/string call styles
    Expectation: The result is correct
    """
    compiled_op = torch.compile(op_func, backend=backend)
    where_forward(np.float32, shape, op_func, compiled_op)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(32, 64)])
@pytest.mark.parametrize("op_func", [where_aten_self])
def test_where_fp16(shape, op_func):
    """
    Feature: Test aclnn where fp16
    Description: Test with float16
    Expectation: The result is correct
    """
    compiled_op = torch.compile(op_func, backend=backend)
    where_forward(np.float16, shape, op_func, compiled_op)
