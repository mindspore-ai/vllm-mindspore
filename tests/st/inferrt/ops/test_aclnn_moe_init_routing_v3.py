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

import torch
import torch_npu
from mrt.torch import backend

import pytest
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_aclnn_moe_init_routing_v3_op(pipeline, monkeypatch):
    """
    Feature: Check aclnnMoeInitRoutingV3 launch
    Description: Check aclnnMoeInitRoutingV3 launch with symbolic infershape
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")

    def moe_init_routing_v3(x, expert_idx, scale_optional, offset_optional, active_num, expert_capacity,
                            expert_num, drop_pad_mode, expert_tokens_num_type, expert_tokens_num_flag,
                            quant_mode, active_expert_range_optional, row_idx_type):
        return torch_npu.npu_moe_init_routing_v2(x, expert_idx, scale=scale_optional, offset=offset_optional,
                                        active_num=active_num, expert_capacity=expert_capacity, expert_num=expert_num,
                                        drop_pad_mode=drop_pad_mode, expert_tokens_num_type=expert_tokens_num_type,
                                        expert_tokens_num_flag=expert_tokens_num_flag, quant_mode=quant_mode,
                                        active_expert_range=active_expert_range_optional, row_idx_type=row_idx_type)

    opt_moe_init_routing_v3 = torch.compile(moe_init_routing_v3, backend=backend)

    bs = 1
    h = 613
    k = 475
    active_num = 475
    expert_capacity = -1
    expert_num = 226
    drop_pad_mode = 0
    expert_tokens_num_type = 1
    expert_tokens_num_flag = True
    quant_mode = -1
    active_expert_range = [23, 35]
    row_idx_type = 0

    x = torch.randn((bs, h), dtype=torch.float32).npu()
    expert_idx = torch.randint(0, expert_num, (bs, k), dtype=torch.int32).npu()
    scale = torch.randn((bs,), dtype=torch.float32).npu()
    offset = None

    expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale =\
        moe_init_routing_v3(x, expert_idx, scale, offset,
                            active_num, expert_capacity,
                            expert_num, drop_pad_mode,
                            expert_tokens_num_type,
                            expert_tokens_num_flag,
                            quant_mode, active_expert_range,
                            row_idx_type)
    opt_expanded_x, opt_expanded_row_idx, opt_expert_tokens_count_or_cumsum, opt_expanded_scale =\
        opt_moe_init_routing_v3(x, expert_idx, scale, offset,
                                active_num, expert_capacity,
                                expert_num, drop_pad_mode,
                                expert_tokens_num_type,
                                expert_tokens_num_flag, quant_mode, active_expert_range, row_idx_type)
    assert torch.equal(expanded_x, opt_expanded_x), f"\nexpanded_x={expanded_x}\nopt_expanded_x={opt_expanded_x}"
    assert torch.equal(expanded_row_idx, opt_expanded_row_idx), \
                       f"\nexpanded_row_idx={expanded_row_idx}\nopt_expanded_row_idx={opt_expanded_row_idx}"
    assert torch.equal(expert_tokens_count_or_cumsum, opt_expert_tokens_count_or_cumsum), \
                       f"\nexpert_tokens_count_or_cumsum={expert_tokens_count_or_cumsum}\n\
                       opt_expert_tokens_count_or_cumsum={opt_expert_tokens_count_or_cumsum}"
    assert torch.equal(expanded_scale, opt_expanded_scale), f"\nexpanded_scale={expanded_scale}\
                       \nopt_expanded_scale={opt_expanded_scale}"
