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
import numpy as np
import torch

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual
from mrt.torch import fx_mlir_backend as backend

def bitwise_and_op(x, y):
    return x & y

def get_op_func_compiled():
    return torch.compile(bitwise_and_op, backend=backend)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
@pytest.mark.parametrize("shape", ([2, 3], [15, 64], [1024, 512]))
@pytest.mark.parametrize("dtype", (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8))
def test_bitwise_and(pipeline, monkeypatch, shape, dtype):
    """
    Feature: Test aclnn bitwise and
    Description: Test aclnn bitwise and with different dtype inputs and different shape
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")

    tensor_x = torch.rand(shape, dtype=dtype, device="npu")
    tensor_y = torch.rand(shape, dtype=dtype, device="npu")

    tensor_x_cpu = tensor_x.cpu()
    tensor_y_cpu = tensor_y.cpu()

    result_operate = bitwise_and_op(tensor_x_cpu, tensor_y_cpu)

    compile_op = get_op_func_compiled()
    result_compile_op = compile_op(tensor_x, tensor_y).cpu()

    AssertRtolEqual(result_operate, result_compile_op)
