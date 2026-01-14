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

import numpy as np
import torch
import torch_npu
from mrt.torch.fx_mlir_backend import backend

import pytest
from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual

def moe_init_routing_golden_shape(x, expert_idx, scale, offset, active_num, expert_capacity,
                                  expert_num, drop_pad_mode, expert_tokens_num_type, expert_tokens_num_flag,
                                  active_expert_range, quant_mode, row_idx_type):
    expert_start = active_expert_range[0] if drop_pad_mode == 0 else 0
    expert_end = active_expert_range[1] if drop_pad_mode == 0 else expert_num
    num_rows = x.shape[0]
    h = x.shape[1]
    k = expert_idx.shape[-1]
    expert_idx_in = expert_idx.copy().reshape(-1)
    actual_expert_total_num = np.sum((expert_idx_in >= expert_start) & (expert_idx_in < expert_end))

    # Calculate expanded_row_idx shape
    if row_idx_type == 1:
        expanded_row_idx_shape = (actual_expert_total_num,)
    else:
        expanded_row_idx_shape = (num_rows * k,)

    # Calculate expert_tokens_count shape
    if not expert_tokens_num_flag:
        expert_tokens_count_shape = None
    else:
        if drop_pad_mode == 0:
            if expert_tokens_num_type == 2:
                # For type 2, it's a 2D array with expert_id and counts
                expert_tokens_count_shape = (expert_num, 2)
            else:
                expert_tokens_count_shape = (expert_end - expert_start,)
        else:
            expert_tokens_count_shape = (expert_end - expert_start,)

    # Calculate expanded_x and expanded_scale shapes
    if drop_pad_mode == 0:
        if active_num == 0:
            active_num = actual_expert_total_num
        else:
            active_num = min(active_num, actual_expert_total_num)
        expanded_x_shape = (active_num, h)
        expanded_scale_shape = None
        if scale is not None and quant_mode == -1:
            expanded_scale_shape = (active_num,)
    else:
        expanded_x_shape = (expert_num, expert_capacity, h)
        expanded_scale_shape = None
        if scale is not None:
            if quant_mode == -1:
                expanded_scale_shape = (expert_num * expert_capacity,)

    return expanded_x_shape, expanded_row_idx_shape, expert_tokens_count_shape, expanded_scale_shape


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
@pytest.mark.parametrize("num_rows", [1, 4])
def test_aclnn_moe_init_routing_v3_op(pipeline, monkeypatch, num_rows):
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

    bs = num_rows
    h = 14
    k = 5
    active_num = k*bs
    expert_capacity = -1
    expert_num = 226
    drop_pad_mode = 0
    expert_tokens_num_type = 1
    expert_tokens_num_flag = True
    quant_mode = -1
    active_expert_range = [0, 8]
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
    
    x_np = x.detach().cpu().numpy()
    expert_idx_np = expert_idx.detach().cpu().numpy()
    scale_np = scale.detach().cpu().numpy()
    offset_np = offset.detach().cpu().numpy() if offset is not None else None
    expanded_x_shape, expanded_row_idx_shape, expert_tokens_count_shape, expanded_scale_shape = \
        moe_init_routing_golden_shape(x_np, expert_idx_np, scale_np, offset_np, active_num,
                                      expert_capacity, expert_num, drop_pad_mode, expert_tokens_num_type,
                                      expert_tokens_num_flag, active_expert_range, quant_mode, row_idx_type)
    x_valid_len = expanded_x_shape[0]
    row_valid_len = expanded_row_idx_shape[0]
    count_valid_len = expert_tokens_count_shape[0] if expert_tokens_count_shape is not None else 0
    scale_valid_len = expanded_scale_shape[0] if expanded_scale_shape is not None else 0
    AssertRtolEqual(expanded_x.detach().cpu()[:x_valid_len], opt_expanded_x.detach().cpu()[:x_valid_len])
    AssertRtolEqual(expanded_row_idx.detach().cpu()[:row_valid_len], opt_expanded_row_idx.detach().cpu()[:row_valid_len])
    AssertRtolEqual(expert_tokens_count_or_cumsum.detach().cpu()[:count_valid_len], opt_expert_tokens_count_or_cumsum.detach().cpu()[:count_valid_len])
    AssertRtolEqual(expanded_scale.detach().cpu()[:scale_valid_len], opt_expanded_scale.detach().cpu()[:scale_valid_len])


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_aclnn_moe_init_routing_v3_op1(pipeline, monkeypatch):
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
    active_num = k*bs
    expert_capacity = -1
    expert_num = 226
    drop_pad_mode = 0
    expert_tokens_num_type = 1
    expert_tokens_num_flag = True
    quant_mode = -1
    active_expert_range = [23, 25]
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

    x_np = x.detach().cpu().numpy()
    expert_idx_np = expert_idx.detach().cpu().numpy()
    scale_np = scale.detach().cpu().numpy()
    offset_np = offset.detach().cpu().numpy() if offset is not None else None
    expanded_x_shape, expanded_row_idx_shape, expert_tokens_count_shape, expanded_scale_shape = \
        moe_init_routing_golden_shape(x_np, expert_idx_np, scale_np, offset_np, active_num,
                                      expert_capacity, expert_num, drop_pad_mode, expert_tokens_num_type,
                                      expert_tokens_num_flag, active_expert_range, quant_mode, row_idx_type)
    x_valid_len = expanded_x_shape[0]
    row_valid_len = expanded_row_idx_shape[0]
    count_valid_len = expert_tokens_count_shape[0] if expert_tokens_count_shape is not None else 0
    scale_valid_len = expanded_scale_shape[0] if expanded_scale_shape is not None else 0
    AssertRtolEqual(expanded_x.detach().cpu()[:x_valid_len], opt_expanded_x.detach().cpu()[:x_valid_len])
    AssertRtolEqual(expanded_row_idx.detach().cpu()[:row_valid_len], opt_expanded_row_idx.detach().cpu()[:row_valid_len])
    AssertRtolEqual(expert_tokens_count_or_cumsum.detach().cpu()[:count_valid_len], opt_expert_tokens_count_or_cumsum.detach().cpu()[:count_valid_len])
    AssertRtolEqual(expanded_scale.detach().cpu()[:scale_valid_len], opt_expanded_scale.detach().cpu()[:scale_valid_len])
