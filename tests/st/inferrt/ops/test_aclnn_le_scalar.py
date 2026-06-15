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
"""Tests for aten.le.Scalar operation."""

import pytest
import torch

from ms_inferrt.torch import backend
from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def aten_le_scalar(x, scalar):
    """Call the exact aten overload requested by model lowering."""
    return torch.ops.aten.le.Scalar(x, scalar)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scalar", [0.5, 2])
def test_aten_le_scalar(dtype, scalar):
    """
    Feature: Test aten.le.Scalar
    Description: Verify exact aten less-equal scalar overload through fx_backend.
    Expectation: The result is correct.
    """
    cpu_input = torch.rand((8, 16), dtype=dtype)
    npu_input = cpu_input.npu()
    cpu_output = aten_le_scalar(cpu_input, scalar)
    op_func_compiled = torch.compile(aten_le_scalar, backend=backend)
    npu_output = op_func_compiled(npu_input, scalar)
    AssertRtolEqual(cpu_output, npu_output.cpu())
