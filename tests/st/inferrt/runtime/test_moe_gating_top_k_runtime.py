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

"""Runtime regression tests for moe_gating_top_k lowering."""
import pytest
import torch

from ms_inferrt.torch.fx_backend import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_c_ascend_moe_gating_top_k_bias_opt_kwargs_runtime(backend):
    """
    Feature: Test _C_ascend moe_gating_top_k lowering
    Description: Verify kwargs-style call with bias_opt/out_flag is lowered
                 and executed correctly under fx backend
    Expectation: Compiled and eager results are consistent
    """
    ascend_ns = getattr(torch.ops, "_C_ascend", None)
    if ascend_ns is None or not hasattr(ascend_ns, "moe_gating_top_k"):
        pytest.skip("torch.ops._C_ascend.moe_gating_top_k is unavailable")

    op = torch.ops._C_ascend.moe_gating_top_k  # pylint: disable=protected-access

    def func(logits):
        return op(
            logits,
            k=8,
            k_group=1,
            group_count=1,
            group_select_mode=1,
            renorm=0,
            norm_type=0,
            out_flag=False,
            routed_scaling_factor=1.0,
            eps=1e-20,
            bias_opt=None,
        )

    compiled_op = torch.compile(func, backend=backend)
    logits = torch.randn(16, 256, dtype=torch.bfloat16).npu()

    eager_y, eager_idx, _ = func(logits)
    compiled_y, compiled_idx, _ = compiled_op(logits)

    AssertRtolEqual(compiled_y, eager_y)
    AssertRtolEqual(compiled_idx, eager_idx)
