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
"""Tests for aclnn var_mean operation (aten.var_mean.correction, torch.var_mean, etc.)."""
import numpy as np
import pytest
import torch

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def _torch_var_mean(dim, correction, keepdim):
    def func(x):
        if correction is None and keepdim is None:
            return torch.var_mean(x, dim=dim)
        if correction is None:
            return torch.var_mean(x, dim=dim, keepdim=keepdim)
        if keepdim is None:
            return torch.var_mean(x, dim=dim, correction=correction)
        return torch.var_mean(x, dim=dim, correction=correction, keepdim=keepdim)
    return func


def _aten_var_mean(dim, correction, keepdim):
    def func(x):
        if correction is None and keepdim is None:
            return torch.ops.aten.var_mean.correction(x, dim=dim)
        if correction is None:
            return torch.ops.aten.var_mean.correction(x, dim=dim, keepdim=keepdim)
        if keepdim is None:
            return torch.ops.aten.var_mean.correction(x, dim=dim, correction=correction)
        return torch.ops.aten.var_mean.correction(x, dim=dim, correction=correction, keepdim=keepdim)
    return func


def var_mean_forward(dtype, shape, op_func, compiled_func):
    """
    var_mean forward function. Returns tuple (mean, var)
    """
    if np.issubdtype(dtype, np.integer):
        cpu_input = np.random.randint(-100, 100, shape).astype(dtype)
    else:
        cpu_input = np.random.uniform(-1, 1, shape).astype(dtype)

    npu_input = torch.from_numpy(cpu_input).npu()

    ori_mean, ori_var = op_func(npu_input)
    npu_mean, npu_var = compiled_func(npu_input)

    ori_mean = ori_mean.detach().cpu().numpy()
    ori_var = ori_var.detach().cpu().numpy()
    npu_mean = npu_mean.detach().cpu().numpy()
    npu_var = npu_var.detach().cpu().numpy()

    AssertRtolEqual(ori_mean, npu_mean)
    AssertRtolEqual(ori_var, npu_var)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(128, 256, 64)])
@pytest.mark.parametrize("dim,correction,keepdim", [
    (-1, None, None),           # only dim (scalar), both defaults
    (-1, 0, None),              # dim (scalar) + correction, omit keepdim
    (-1, None, True),           # dim (scalar) + keepdim, omit correction
    (-1, 0, True),              # all explicit
    (None, None, None),         # dim=None (expand to all), both defaults
    ([0, 1], 0, False),         # dim=list, explicit correction/keepdim
])
@pytest.mark.parametrize("op_builder", [_torch_var_mean, _aten_var_mean])
# pylint: disable=redefined-outer-name
def test_var_mean_partial_defaults(shape, dim, correction, keepdim, op_builder):
    """
    Feature: var_mean_arg_hook default value completion (partial omission)
    Description: torch/aten style with one or both of correction/keepdim omitted
    Expectation: hook fills correction=1/keepdim=False correctly
    """
    op_func = op_builder(dim, correction, keepdim)
    compiled_op = torch.compile(op_func, backend=backend)
    var_mean_forward(np.float32, shape, op_func, compiled_op)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(32, 64)])
@pytest.mark.parametrize("dim,correction,keepdim", [
    (-1, None, None),
])
@pytest.mark.parametrize("op_builder", [_torch_var_mean])
def test_var_mean_fp16(shape, dim, correction, keepdim, op_builder):
    """
    Feature: var_mean fp16 with hook defaults
    Description: basic coverage with float16
    Expectation: result matches reference
    """
    op_func = op_builder(dim, correction, keepdim)
    compiled_op = torch.compile(op_func, backend=backend)
    var_mean_forward(np.float16, shape, op_func, compiled_op)
