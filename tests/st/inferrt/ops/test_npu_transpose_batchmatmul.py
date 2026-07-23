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
"""Tests for torch.ops.npu.npu_transpose_batchmatmul via InferRT."""
import pytest
import torch

from ms_inferrt.ir import Op
from ms_inferrt.torch import backend
from ms_inferrt.torch import fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def _npu_transpose_batchmatmul(input_tensor, weight, perm_x1=(1, 0, 2), batch_split_factor=1):
    return torch.ops.npu.npu_transpose_batchmatmul(
        input_tensor,
        weight,
        bias=None,
        scale=None,
        perm_x1=perm_x1,
        perm_x2=(0, 1, 2),
        perm_y=(1, 0, 2),
        batch_split_factor=batch_split_factor,
    )


def _assert_close_to_eager(input_tensor, weight, perm_x1=(1, 0, 2), batch_split_factor=1):
    """Compare compiled npu_transpose_batchmatmul output against eager execution."""
    expected = _npu_transpose_batchmatmul(
        input_tensor,
        weight,
        perm_x1=perm_x1,
        batch_split_factor=batch_split_factor,
    )
    compiled_func = torch.compile(_npu_transpose_batchmatmul, backend=backend, fullgraph=True)
    actual = compiled_func(
        input_tensor,
        weight,
        perm_x1=perm_x1,
        batch_split_factor=batch_split_factor,
    )
    AssertRtolEqual(expected.detach().cpu(), actual.detach().cpu())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_npu_transpose_batchmatmul_lowering_uses_builtin_op():
    """
    Feature: npu_transpose_batchmatmul lowering
    Description: Verify the NPU frontend target maps to a built-in InferRT op instead of custom_call.
    Expectation: The target maps to Op.npu_transpose_batchmatmul.
    """
    # pylint: disable=protected-access
    assert fx_backend._get_op(torch.ops.npu.npu_transpose_batchmatmul) == Op.npu_transpose_batchmatmul
    assert fx_backend._get_op("npu.npu_transpose_batchmatmul") == Op.npu_transpose_batchmatmul


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_transpose_batchmatmul_matches_eager(dtype):
    """
    Feature: npu_transpose_batchmatmul execution
    Description: Compare InferRT output with torch_npu eager output for DSv4 attention-style permutations.
    Expectation: The result is correct.
    """
    torch.manual_seed(0)
    input_tensor = torch.randn((16, 4, 128), dtype=dtype).npu()
    weight = torch.randn((4, 128, 128), dtype=dtype).npu()
    _assert_close_to_eager(input_tensor, weight)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_npu_transpose_batchmatmul_default_perm_x1_matches_eager():
    """
    Feature: npu_transpose_batchmatmul default perm_x1
    Description: Verify default perm_x1=[0,1,2] is forwarded to aclnnTransposeBatchMatMul.
    Expectation: The result is correct.
    """
    torch.manual_seed(1)
    input_tensor = torch.randn((4, 16, 128), dtype=torch.float16).npu()
    weight = torch.randn((4, 128, 128), dtype=torch.float16).npu()
    expected = torch.ops.npu.npu_transpose_batchmatmul(input_tensor, weight)

    def func(x, w):
        return torch.ops.npu.npu_transpose_batchmatmul(x, w)

    compiled_func = torch.compile(func, backend=backend, fullgraph=True)
    actual = compiled_func(input_tensor, weight)
    AssertRtolEqual(expected.detach().cpu(), actual.detach().cpu())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_npu_transpose_batchmatmul_batch_split_matches_eager():
    """
    Feature: npu_transpose_batchmatmul batch_split_factor
    Description: Verify non-default batch_split_factor output layout.
    Expectation: The result is correct.
    """
    torch.manual_seed(2)
    input_tensor = torch.randn((16, 4, 128), dtype=torch.float16).npu()
    weight = torch.randn((4, 128, 128), dtype=torch.float16).npu()
    _assert_close_to_eager(input_tensor, weight, batch_split_factor=2)
