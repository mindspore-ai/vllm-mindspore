# Copyright 2025 Huawei Technologies Co., Ltd
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

import pytest
import torch

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual
from mrt.torch import backend


def op_func(input_tensor):
    return input_tensor.contiguous()


def get_op_func_compiled():
    def custom_op_func(input_tensor):
        return input_tensor.contiguous()
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
@pytest.mark.parametrize("shape", [[10, 40], [20, 30, 35]])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_contiguous(pipeline, monkeypatch, shape, dtype):
    """
    Feature: Test tensor.contiguous
    Description: Test contiguous with dtype inputs
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")

    input_tensor = torch.rand(shape, dtype=dtype)
    input_tensor = input_tensor.transpose(1, 0)
    input_tensor_npu = input_tensor.npu()
    cpu_output0 = op_func(input_tensor)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(input_tensor_npu)
    AssertRtolEqual(cpu_output0, npu_output.detach().cpu())
