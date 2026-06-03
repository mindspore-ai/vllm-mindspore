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
"""Tests for aten.mean via fx_backend."""
import numpy as np
import torch

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def get_mean_compiled():
    def custom_op_func(x, dim, keepdim):
        return torch.mean(x, dim=dim, keepdim=keepdim)
    return torch.compile(custom_op_func, backend=backend)


def get_pow_mean_compiled():
    def custom_op_func(x):
        return x.pow(2).mean(dim=-1, keepdim=True)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_mean():
    """
    Feature: Test aten.mean via fx_backend
    Description: reduce mean along dim
    Expectation: Result matches reference
    """
    cpu_x = np.random.uniform(-1, 1, [32, 128]).astype(np.float32)
    npu_x = torch.from_numpy(cpu_x).npu()
    op_func_compiled = get_mean_compiled()
    npu_out = op_func_compiled(npu_x, 1, False).detach().cpu().numpy()
    cpu_out = np.mean(cpu_x, axis=1, keepdims=False)
    AssertRtolEqual(cpu_out, npu_out)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_pow_mean():
    """
    Feature: Test chained pow(2).mean(dim=-1, keepdim=True) via fx_backend
    Description: variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    Expectation: Result matches reference
    """
    cpu_x = np.random.uniform(-1, 1, [32, 128]).astype(np.float32)
    npu_x = torch.from_numpy(cpu_x).npu()
    op_func_compiled = get_pow_mean_compiled()
    npu_out = op_func_compiled(npu_x).detach().cpu().numpy()
    cpu_out = np.mean(np.power(cpu_x, 2), axis=-1, keepdims=True)
    AssertRtolEqual(cpu_out, npu_out)
