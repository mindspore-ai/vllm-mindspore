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

"""Tests for aclnn moe_distribute_combine_v2 operation."""

from types import SimpleNamespace

import pytest
import torch

from ms_inferrt.torch.fx_mlir_backend import backend
from ms_inferrt.torch.fx_backend import moe_distribute_combine_v2_hook
from tests.mark_utils import arg_mark


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_moe_distribute_combine_v2_op_registration():
    """
    Feature: Check moe_distribute_combine_v2 op registration
    Description: Ensure torch.compile can build a graph with backend for the op
    Expectation: Compilation succeeds
    """

    assert hasattr(torch.ops, "npu"), "torch.ops.npu not available"
    assert hasattr(
        torch.ops.npu, "npu_moe_distribute_combine_v2"
    ), "npu_moe_distribute_combine_v2 is not registered"

    def moe_combine_test_func():
        pass

    opt_func = torch.compile(moe_combine_test_func, backend=backend)
    assert opt_func is not None, "Failed to compile with backend"


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="dryrun",
    essential_mark="essential",
)
def test_moe_distribute_combine_v2_hook_maps_performance_info():
    """
    Feature: Check moe_distribute_combine_v2 arg mapping
    Description: Verify performance_info is forwarded before communication attributes
    Expectation: The backend schema order is correct
    """

    kwargs = {
        "expand_x": "expand_x",
        "expert_ids": "expert_ids",
        "assist_info_for_combine": "assist_info_for_combine",
        "ep_send_counts": "ep_send_counts",
        "expert_scales": "expert_scales",
        "tp_send_counts": "tp_send_counts",
        "x_active_mask": "x_active_mask",
        "expand_scales": "expand_scales",
        "shared_expert_x": "shared_expert_x",
        "elastic_info": "elastic_info",
        "ori_x": "ori_x",
        "const_expert_alpha_1": "const_expert_alpha_1",
        "const_expert_alpha_2": "const_expert_alpha_2",
        "const_expert_v": "const_expert_v",
        "performance_info": "performance_info",
        "group_ep": "group_ep",
        "ep_world_size": 8,
        "ep_rank_id": 3,
        "moe_expert_num": 16,
        "group_tp": "group_tp",
        "tp_world_size": 2,
        "tp_rank_id": 1,
        "expert_shard_type": 0,
        "shared_expert_num": 1,
        "shared_expert_rank_num": 1,
        "global_bs": 64,
        "comm_quant_mode": 0,
        "comm_alg": "comm_alg",
        "zero_expert_num": 0,
        "copy_expert_num": 0,
        "const_expert_num": 0,
    }
    node = SimpleNamespace(args=(), kwargs=kwargs)

    mapped_args = moe_distribute_combine_v2_hook(node, [], None)

    assert mapped_args[13] == "const_expert_v"
    assert mapped_args[14] == "performance_info"
    assert mapped_args[15] == "group_ep"
    assert mapped_args[18] == 16
    assert mapped_args[19] == "group_tp"
    assert mapped_args[30] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
