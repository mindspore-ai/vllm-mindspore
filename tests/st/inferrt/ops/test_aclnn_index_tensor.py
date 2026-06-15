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
"""Tests for aten.index.Tensor operation."""

import pytest
import torch

from ms_inferrt.torch import backend
from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def aten_index_tensor(x, indices):
    """Call the exact aten overload requested by model lowering."""
    return torch.ops.aten.index.Tensor(x, indices)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "shape,indices",
    [
        ((8, 4), [torch.tensor([0, 2, 5], dtype=torch.int64)]),
        (
            (4, 5, 3),
            [
                torch.tensor([0, 2, 3], dtype=torch.int64),
                torch.tensor([1, 4, 0], dtype=torch.int64),
                None,
            ],
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_aten_index_tensor(shape, indices, dtype):
    """
    Feature: Test aten.index.Tensor
    Description: Verify exact aten index tensor overload through fx_backend.
    Expectation: The result is correct.
    """
    cpu_input = torch.rand(shape, dtype=dtype)
    npu_input = cpu_input.npu()
    npu_indices = [
        None if index is None else index.npu()
        for index in indices
    ]
    cpu_output = aten_index_tensor(cpu_input, indices)
    op_func_compiled = torch.compile(aten_index_tensor, backend=backend)
    npu_output = op_func_compiled(npu_input, npu_indices)
    AssertRtolEqual(cpu_output, npu_output.cpu())
