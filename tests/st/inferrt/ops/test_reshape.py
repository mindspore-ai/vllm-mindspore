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
from torch_npu.testing.common_utils import create_common_tensor

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual
from mrt.torch import fx_mlir_backend as backend


def op_func(input, shape):
    """op function for reshape"""
    return input.reshape(shape)

def reshape_forward(shape_format, op_func_compiled):
    """
    reshape forward function
    Args:
        shape_format: list of [dtype, format, shape]
        op_func_compiled: The compiled op function.
    """
    for item in shape_format:
        cpu_input, npu_input = create_common_tensor(item, 0, 100)
        shape = [4, 16]
        cpu_output = op_func(cpu_input, shape).detach().numpy()
        npu_output = op_func_compiled(npu_input, shape).detach().cpu().numpy()
        AssertRtolEqual(cpu_output, npu_output)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_reshape(pipeline, monkeypatch):
    """
    Feature: Test reshape
    Description: Test reshape op with mlir_backend
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    dtype_list = [np.float16, np.float32, np.int32, np.bool_]
    format_list = [0]
    shape_list = [[8, 8], [2, 4, 8], [2, 4, 4, 2]]
    
    shape_format = [
        [i, j, k] for i in dtype_list for j in format_list for k in shape_list
    ]
    op_func_compiled = torch.compile(op_func, backend=backend)
    reshape_forward(shape_format, op_func_compiled)
