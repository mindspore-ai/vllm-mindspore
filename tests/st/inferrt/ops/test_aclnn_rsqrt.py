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
"""Tests for aten.rsqrt / torch.rsqrt via fx_backend."""
import numpy as np
import torch

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def get_rsqrt_compiled():
    def custom_op_func(x):
        return torch.rsqrt(x)
    return torch.compile(custom_op_func, backend=backend)


def get_rsqrt_method_compiled():
    def custom_op_func(x):
        return x.rsqrt()
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_rsqrt():
    """
    Feature: Test rsqrt via fx_backend
    Description: reciprocal sqrt
    Expectation: Result matches reference
    """
    cpu_x = np.random.uniform(0.1, 4.0, [64, 64]).astype(np.float32)
    npu_x = torch.from_numpy(cpu_x).npu()
    op_func_compiled = get_rsqrt_compiled()
    npu_out = op_func_compiled(npu_x).detach().cpu().numpy()
    cpu_out = 1.0 / np.sqrt(cpu_x)
    AssertRtolEqual(cpu_out, npu_out)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_rsqrt_method():
    """
    Feature: Test tensor.rsqrt() method via fx_backend
    Description: x.rsqrt() tensor method form (call_method[target=rsqrt])
    Expectation: Result matches reference
    """
    cpu_x = np.random.uniform(0.1, 4.0, [64, 64]).astype(np.float32)
    npu_x = torch.from_numpy(cpu_x).npu()
    op_func_compiled = get_rsqrt_method_compiled()
    npu_out = op_func_compiled(npu_x).detach().cpu().numpy()
    cpu_out = 1.0 / np.sqrt(cpu_x)
    AssertRtolEqual(cpu_out, npu_out)
