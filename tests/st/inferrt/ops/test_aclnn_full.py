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
"""Tests for aten.full.default operation."""

import pytest
import torch

from ms_inferrt.torch import backend
from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def aten_full_default(size, fill_value, dtype, device):
    """Call the exact aten overload requested by model lowering."""
    return torch.ops.aten.full.default(size, fill_value, dtype=dtype, device=device)


def initialize_npu_allocator(dtype):
    """Initialize torch-npu allocator before running a factory-only compiled graph."""
    aten_full_default((1,), 0, dtype, torch.device("npu"))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(2, 3), (4,)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_aten_full_default(shape, dtype):
    """
    Feature: Test aten.full.default
    Description: Verify exact aten full overload through fx_backend.
    Expectation: The result is correct.
    """
    fill_value = 2.5
    initialize_npu_allocator(dtype)
    compiled = torch.compile(aten_full_default, backend=backend)

    expected = aten_full_default(shape, fill_value, dtype, torch.device("cpu"))
    actual = compiled(shape, fill_value, dtype, torch.device("npu")).cpu()

    AssertRtolEqual(expected, actual)
