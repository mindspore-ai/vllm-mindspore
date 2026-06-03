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
"""Tests for aten.tril via fx_backend."""
import numpy as np
import torch

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def get_tril_compiled():
    def custom_op_func(input_tensor):
        return torch.tril(input_tensor)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_tril():
    """
    Feature: Test aten.tril via fx_backend
    Description: Basic tril test
    Expectation: Result matches torch reference
    """
    torch.ones(1, device="npu")

    op_func_compiled = get_tril_compiled()
    input_t = torch.ones(2, 2, device="npu")
    npu_out = op_func_compiled(input_t).detach().cpu().numpy()
    cpu_out = np.tril(np.ones((2, 2), dtype=np.float32))
    AssertRtolEqual(cpu_out, npu_out)
