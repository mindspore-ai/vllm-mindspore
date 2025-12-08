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
"""torch.transpose case"""

import pytest
import numpy as np
import torch

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual
from mrt.torch import fx_mlir_backend as backend


def op_func(input, dim0 ,dim1):
    """golden"""
    return input.transpose(dim0, dim1)


def get_op_func_compiled():
    """transpose op"""
    def custom_op_func(x, dim0, dim1):
        return torch.transpose(x, dim0, dim1)
    return torch.compile(custom_op_func, backend=backend)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("infos", [([2, 3, 4, 9, 3, 2, 5], 6, 2)])
@pytest.mark.parametrize("dtypes", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("pipeline", (True, False))
def test_transpose(infos, dtypes, pipeline, monkeypatch):
    """
    Feature: Test aclnn transpose
    Description: Test aclnn transpose with fp32 inputs
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    cpu_input0 = torch.rand(infos[0], dtype=dtypes)
    npu_input0 = cpu_input0.npu()
    dim0 = infos[1]
    dim1 = infos[2]
    cpu_output = op_func(cpu_input0, dim0, dim1)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(npu_input0, dim0, dim1)
    npu_output_cpu = npu_output.cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)
