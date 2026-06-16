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
"""Tests for aten.softmax via fx_backend."""
import numpy as np
import torch

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def get_softmax_compiled():
    def custom_op_func(x, dim):
        return torch.softmax(x, dim=dim)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_softmax():
    """
    Feature: Test aten.softmax via fx_backend
    Description: softmax along last dimension
    Expectation: Result matches reference
    """
    cpu_x = np.random.uniform(-1, 1, [32, 128]).astype(np.float32)
    npu_x = torch.from_numpy(cpu_x).npu()
    op_func_compiled = get_softmax_compiled()
    npu_out = op_func_compiled(npu_x, -1).detach().cpu().numpy()
    cpu_out = np.exp(cpu_x - np.max(cpu_x, axis=-1, keepdims=True))
    cpu_out = cpu_out / np.sum(cpu_out, axis=-1, keepdims=True)
    AssertRtolEqual(cpu_out, npu_out)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_aten_softmax_default():
    """
    Feature: Test aten._softmax.default via fx_backend
    Description: Directly call torch.ops.aten._softmax.default
    Expectation: Result matches reference
    """
    cpu_x = np.random.uniform(-1, 1, [1, 8, 70, 70]).astype(np.float32)
    npu_x = torch.from_numpy(cpu_x).npu()

    def _softmax_impl(x):
        # pylint: disable=protected-access
        return torch.ops.aten._softmax.default(x, -1, False)

    op_func_compiled = torch.compile(_softmax_impl, backend=backend)
    npu_out = op_func_compiled(npu_x).detach().cpu().numpy()
    cpu_out = np.exp(cpu_x - np.max(cpu_x, axis=-1, keepdims=True))
    cpu_out = cpu_out / np.sum(cpu_out, axis=-1, keepdims=True)
    AssertRtolEqual(cpu_out, npu_out)
