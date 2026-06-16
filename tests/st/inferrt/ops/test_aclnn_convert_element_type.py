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
"""Tests for prims.convert_element_type.default via fx_backend (maps to cast)."""
import numpy as np
import pytest
import torch

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def convert_element_type_aten(x, dtype):
    """torch.ops.prims.convert_element_type.default style"""
    return torch.ops.prims.convert_element_type.default(x, dtype)


def convert_element_type_torch(x, dtype):
    """torch.tensor.to style (exercises cast path)"""
    return x.to(dtype)


def convert_element_type_string(x, dtype):
    """string 'to' method"""
    return x.to(dtype)


def convert_forward(dtype_in, dtype_out, shape, op_func, compiled_func):
    cpu_input = np.random.uniform(-1, 1, shape).astype(dtype_in)
    npu_input = torch.from_numpy(cpu_input).npu()

    ori_output = op_func(npu_input, dtype_out).detach().cpu().numpy()
    npu_output = compiled_func(npu_input, dtype_out).detach().cpu().numpy()

    AssertRtolEqual(ori_output, npu_output)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(128, 256)])
@pytest.mark.parametrize("op_func", [
    convert_element_type_aten, convert_element_type_torch, convert_element_type_string
])
@pytest.mark.parametrize("dtype_in,dtype_out", [
    (np.float32, torch.float16),
    (np.float16, torch.float32),
    (np.float32, torch.int32),
])
# pylint: disable=redefined-outer-name
def test_convert_element_type_fp32(shape, op_func, dtype_in, dtype_out):
    """
    Feature: Test prims.convert_element_type.default (cast) via fx_backend
    Description: Test prims.convert_element_type.default and cast path with fp32/fp16 inputs
    Expectation: The result is correct for aten/torch/string call styles
    """
    compiled_op = torch.compile(op_func, backend=backend)
    convert_forward(dtype_in, dtype_out, shape, op_func, compiled_op)
