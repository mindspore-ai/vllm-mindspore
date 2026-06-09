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
"""Tests for aten.exp.default operation."""

import pytest
import torch

from ms_inferrt.torch import backend
from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def aten_exp_default(x):
    """Call the exact aten overload requested by model lowering."""
    return torch.ops.aten.exp.default(x)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(2, 3), (16, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_aten_exp_default(shape, dtype):
    """
    Feature: Test aten.exp.default
    Description: Verify exact aten exp overload through fx_backend.
    Expectation: The result is correct.
    """
    cpu_input = torch.randn(shape, dtype=dtype).clamp(min=-3, max=3)
    npu_input = cpu_input.npu()

    compiled = torch.compile(aten_exp_default, backend=backend)
    expected = aten_exp_default(cpu_input)
    actual = compiled(npu_input).cpu()

    prec = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    AssertRtolEqual(expected, actual, prec)
