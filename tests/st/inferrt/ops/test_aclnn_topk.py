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
"""Tests for aten.topk via fx_backend."""
import torch

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def get_topk_compiled():
    def custom_op_func(input_tensor, k, dim):
        return torch.topk(input_tensor, k, dim=dim)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_topk():
    """
    Feature: Test aten.topk via fx_backend
    Description: Basic topk test
    Expectation: Result matches torch reference
    """
    torch.ones(1, device="npu")

    op_func_compiled = get_topk_compiled()
    input_t = torch.tensor([[1.0, 3.0, 2.0, 4.0]], device="npu")
    npu_values, npu_indices = op_func_compiled(input_t, 2, -1)
    npu_values = npu_values.detach().cpu().numpy()
    npu_indices = npu_indices.detach().cpu().numpy()
    cpu_values, cpu_indices = torch.topk(torch.tensor([[1.0, 3.0, 2.0, 4.0]]), 2, dim=-1)
    cpu_values = cpu_values.numpy()
    cpu_indices = cpu_indices.numpy()
    AssertRtolEqual(cpu_values, npu_values)
    AssertRtolEqual(cpu_indices, npu_indices)
