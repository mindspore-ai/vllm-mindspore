#!/usr/bin/env python
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
"""
Tests for stack operation on Ascend using MRT backends.

This module compares torch.stack with the compiled implementation
lowered to the Ascend ACLNN stack kernel.
"""

import pytest
import torch

from ms_inferrt.torch import backend
from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(inputs, dim):
    """Golden reference implemented with torch.stack."""
    return torch.stack(inputs, dim=dim)


def get_op_func_compiled():
    """Return compiled stack op using MRT backend."""

    def custom_op_func(inputs, dim):
        return torch.stack(inputs, dim=dim)

    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "shape,dim,dtype",
    [
        ((4, 8), 0, torch.float16),
        ((4, 8), 1, torch.float16),
        ((2, 3, 5), 0, torch.bfloat16),
        ((2, 3, 5), -1, torch.bfloat16),
        ((1, 4, 6), 2, torch.float32),
    ],
)
@pytest.mark.parametrize("num_tensors", [2, 3])
def test_stack(shape, dim, dtype, num_tensors):
    """
    Feature: Test aclnn stack
    Description: Compare compiled stack with eager torch.stack
    Expectation: Results match within reasonable tolerance
    """
    cpu_inputs = [torch.randn(shape, dtype=dtype) for _ in range(num_tensors)]
    npu_inputs = [x.npu() for x in cpu_inputs]

    cpu_out = op_func(cpu_inputs, dim)

    compiled = get_op_func_compiled()
    npu_out = compiled(npu_inputs, dim).cpu()

    AssertRtolEqual(cpu_out, npu_out)
