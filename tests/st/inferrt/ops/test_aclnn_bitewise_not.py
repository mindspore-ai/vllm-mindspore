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
"""Tests for aclnn bitwise not operation."""
import pytest
import torch

from mrt.torch import fx_mlir_backend as backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual

def bitwise_not_op(x):
    return ~x

def get_op_func_compiled():
    return torch.compile(bitwise_not_op, backend=backend)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", ([2, 3], [15, 64], [1024, 512]))
@pytest.mark.parametrize("dtype", (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8))
def test_bitwise_not(shape, dtype):
    """
    Feature: Test aclnn bitwise not
    Description: Test aclnn bitwise not with different dtype inputs and different shape
    Expectation: The result is correct
    """

    tensor_x = torch.rand(shape, dtype=dtype, device="npu")

    tensor_x_cpu = tensor_x.cpu()

    result_operate = bitwise_not_op(tensor_x_cpu)

    compile_op = get_op_func_compiled()
    result_compile_op = compile_op(tensor_x).cpu()

    AssertRtolEqual(result_operate, result_compile_op)
