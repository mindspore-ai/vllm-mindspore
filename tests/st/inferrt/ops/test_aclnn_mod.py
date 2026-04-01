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
"""Tests for aclnn mod operation."""
import pytest
import torch

from ms_inferrt.torch import fx_mlir_backend as backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual

def mod_op(x, y):
    return x % y

def get_op_func_compiled():
    return torch.compile(mod_op, backend=backend)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", ([2, 3], [512, 256]))
def test_mod(shape):
    """
    Feature: Test aclnn mod
    Description: Test aclnn mod and with different dtype inputs and different shape
    Expectation: The result is correct
    """

    tensor_x = torch.rand(shape, dtype=torch.float16, device="npu")
    tensor_y = torch.rand(shape, dtype=torch.float16, device="npu")

    tensor_x_cpu = tensor_x.cpu()
    tensor_y_cpu = tensor_y.cpu()

    result_operate = mod_op(tensor_x_cpu, tensor_y_cpu)

    compile_op = get_op_func_compiled()
    result_compile_op = compile_op(tensor_x, tensor_y).cpu()

    print(f"result_operate: {result_operate}")
    print(f"result_compile_op: {result_compile_op}")
    AssertRtolEqual(result_operate, result_compile_op)
