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
"""Tests for aten.arange via fx_backend."""
import numpy as np
import torch

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def get_arange_compiled():
    def custom_op_func(start, end, step):
        return torch.arange(start, end, step, device="npu")
    return torch.compile(custom_op_func, backend=backend)


def get_arange_no_step_compiled():
    def custom_op_func(start, end):
        return torch.arange(start, end, device="npu")
    return torch.compile(custom_op_func, backend=backend)


def get_arange_end_only_compiled():
    def custom_op_func(end):
        return torch.arange(end, device="npu", dtype=torch.int64)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_arange():
    """
    Feature: Test aten.arange via fx_backend
    Description: Basic arange test
    Expectation: Result matches torch reference
    """
    torch.ones(1, device="npu")

    op_func_compiled = get_arange_compiled()
    npu_out = op_func_compiled(0, 10, 1).detach().cpu().numpy()
    cpu_out = np.arange(0, 10, 1, dtype=np.int64)
    AssertRtolEqual(cpu_out, npu_out)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_arange_no_step():
    """
    Feature: Test aten.arange via fx_backend without step
    Description: arange without explicit step (default step=1)
    Expectation: Result matches torch reference
    """
    torch.ones(1, device="npu")

    op_func_compiled = get_arange_no_step_compiled()
    npu_out = op_func_compiled(0, 10).detach().cpu().numpy()
    cpu_out = np.arange(0, 10, dtype=np.int64)
    AssertRtolEqual(cpu_out, npu_out)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_arange_end_only():
    """
    Feature: Test aten.arange via fx_backend with only end argument
    Description: arange(end, device=..., dtype=...) form (single positional arg)
    Expectation: Result matches torch reference
    """
    torch.ones(1, device="npu")

    op_func_compiled = get_arange_end_only_compiled()
    npu_out = op_func_compiled(29).detach().cpu().numpy()
    cpu_out = np.arange(29, dtype=np.int64)
    AssertRtolEqual(cpu_out, npu_out)
