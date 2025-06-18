# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2025 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fused MoE kernel with MindSpore."""

from typing import Optional

import mindspore as ms
import numpy as np
from mindspore import Tensor, mint, nn, ops
from mindspore.ops.auto_generate import (GroupedMatmulV4, MoeDistributeCombine,
                                         MoeDistributeDispatch,
                                         MoeGatingTopKSoftmax,
                                         MoeInitRoutingV2, MoeTokenUnpermute)
from vllm.distributed.parallel_state import get_ep_group

from vllm_mindspore.model_executor.layers.fused_moe.config import MoeMode
from vllm_mindspore.utils import is_910b


def fused_topk(
    hidden_states: Tensor,
    gating_output: Tensor,
    topk: int,
    renormalize: bool,
    indices_type=None,
) -> tuple[Tensor, Tensor]:
    moe_topk_softmax = MoeGatingTopKSoftmax()
    topk_weights, topk_ids, _ = moe_topk_softmax(gating_output, None, topk)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if indices_type is not None:
        topk_ids = topk_ids.to(indices_type)
    return topk_weights, topk_ids


def grouped_topk(
        hidden_states: Tensor,
        gating_output: Tensor,
        topk: int,
        renormalize: bool,
        num_expert_group: int = 0,
        topk_group: int = 0,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[Tensor] = None
) -> tuple[Tensor, Tensor]:
    raise NotImplementedError("grouped_topk is not implemented.")


class FusedExperts(nn.Cell):

    def __init__(self, moe_config):
        super().__init__()
        self.group_matmul_ops = GroupedMatmulV4()
        self.moe_init_routing_op = MoeInitRoutingV2()
        self.moe_token_unpermute = MoeTokenUnpermute()

        self.moe_mode = None

        self.experts_num = moe_config.num_experts
        self.local_expert_num = moe_config.num_local_experts
        self.ep_size = moe_config.moe_parallel_config.ep_size
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        self.dp_size = moe_config.moe_parallel_config.dp_size
        if self.ep_size > 1:
            experts_num_map = [(self.experts_num // self.ep_size)
                               for _ in range(self.ep_size - 1)]
            experts_num_map.append(self.experts_num -
                                   ((self.experts_num // self.ep_size) *
                                    (self.ep_size - 1)))
            self.experts_num_map = experts_num_map
            self.ep_group = get_ep_group().device_group._name

        # pure ep mode
        if moe_config.moe_parallel_config.ep_size > 1 and \
           moe_config.moe_parallel_config.tp_size == 1:
            self.moe_mode = MoeMode.PURE_EP
            self.dispatch = MoeDistributeDispatch()
            self.combine = MoeDistributeCombine()
            self.dispatch_tp_world_size = 0 if is_910b() else 1
            self.dispatch_shared_expert_num = 0 if is_910b() else 1
            self.max_bs = 256 if is_910b() else 512
            self.max_bs *= self.ep_size

        # pure tp mode
        elif moe_config.moe_parallel_config.ep_size == 1 and \
           moe_config.moe_parallel_config.tp_size >= 1:
            self.moe_mode = MoeMode.PURE_TP
        # tp + ep mode
        else:
            self.moe_mode = MoeMode.TP_MIX_EP
            experts_num_map_np = np.array(self.experts_num_map, dtype=np.int32)
            experts_num_map_cu_np = np.cumsum(experts_num_map_np,
                                              dtype=np.int32)
            # the start index of experts for current ep rank
            self.expert_start_index = 0 if self.ep_rank == 0 else int(
                experts_num_map_cu_np[self.ep_rank - 1])

    def construct(self,
                  hidden_states: Tensor,
                  w1: Tensor,
                  w2: Tensor,
                  topk_weights: Tensor,
                  topk_ids: Tensor,
                  activation: str = "silu",
                  global_num_experts: int = -1,
                  apply_router_weight_on_input: bool = False) -> Tensor:
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "apply_router_weight_on_input is not implemented.")

        if self.moe_mode == MoeMode.PURE_TP:
            hidden_states = self.run_tp(hidden_states, w1, w2, topk_ids,
                                        topk_weights, activation,
                                        global_num_experts)
        elif self.moe_mode == MoeMode.PURE_EP:
            hidden_states = self.run_ep(hidden_states, w1, w2, topk_ids,
                                        topk_weights, activation,
                                        global_num_experts)
        elif self.moe_mode == MoeMode.TP_MIX_EP:
            hidden_states = self.run_tp_mix_ep(hidden_states, w1, w2, topk_ids,
                                               topk_weights, activation,
                                               global_num_experts)
        else:
            raise ValueError(f"Unsupported moe mode: {self.moe_mode}")

        return hidden_states

    def _gate_activation(self, gate, activation):
        if activation == "silu":
            return mint.nn.functional.silu(gate)
        elif activation == "gelu":
            return mint.nn.functional.gelu(gate)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def _group_matmul(self, hidden_states, weight, group_list):
        return self.group_matmul_ops([hidden_states], [weight],
                                     None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     group_list,
                                     split_item=3,
                                     group_type=0,
                                     group_list_type=1)[0]

    def _ffn(self, hidden_state, w1, w2, group_list, activation):
        gate_hidden_out = self._group_matmul(hidden_state, w1, group_list)
        gate, hidden = mint.split(gate_hidden_out,
                                  (w1.shape[2] // 2, w1.shape[2] // 2), -1)
        gate = self._gate_activation(gate, activation)
        hidden = mint.mul(hidden, gate)
        expert_output = self._group_matmul(hidden, w2, group_list)
        expert_output = mint.nan_to_num(expert_output, 0, 0, 0)
        return expert_output

    def run_tp(self, hidden_states, w1, w2, topk_ids, topk_weights, activation,
               global_num_experts):
        topk_weights = topk_weights.astype(hidden_states.dtype)
        topk_ids = topk_ids.astype(ms.int32)

        sorted_input_tensor, unsort_map, group_list, _ = \
            self.moe_init_routing_op(
                hidden_states,
                topk_ids,
                active_num=0,
                expert_capacity=0,
                expert_num=global_num_experts,
                drop_pad_mode=0,
                expert_tokens_count_or_cumsum_flag=2,
                expert_tokens_before_capacity_flag=True)

        group_list = group_list.astype(ms.int64)

        expert_output = self._ffn(sorted_input_tensor, w1, w2, group_list,
                                  activation)

        moe_output = self.moe_token_unpermute(permuted_tokens=expert_output,
                                              sorted_indices=unsort_map,
                                              probs=topk_weights,
                                              padded_mode=False,
                                              restore_shape=None)
        return moe_output

    def run_tp_mix_ep(self, hidden_states, w1, w2, topk_ids, topk_weights,
                      activation, global_num_experts):
        topk_weights = topk_weights.astype(hidden_states.dtype)
        topk_ids = topk_ids.astype(ms.int32)

        topk_mask = topk_ids < self.expert_start_index
        local_topk_ids = topk_ids - self.expert_start_index
        local_topk_ids = local_topk_ids.astype(ms.int32)
        # trick: if tp + ep moe, means ep_size > 1,
        # and expert will be distributed across ep_size,
        # so except last ep rank, max(local_topk_ids) self.experts_num - 1.
        # It will allow ffn not compute the expert output,
        # which are not assigned to this ep rank.
        local_topk_ids = ops.masked_fill(local_topk_ids, topk_mask,
                                         self.experts_num - 1)

        weight_mask = local_topk_ids >= self.local_expert_num
        topk_weights = ops.masked_fill(topk_weights, weight_mask, 0)

        sorted_input_tensor, unsort_map, group_list, _ = \
            self.moe_init_routing_op(
                hidden_states,
                local_topk_ids,
                active_num=0,
                expert_capacity=0,
                expert_num=global_num_experts,
                drop_pad_mode=0,
                expert_tokens_count_or_cumsum_flag=2,
                expert_tokens_before_capacity_flag=True)

        group_list = group_list[:self.local_expert_num]
        group_list = group_list.astype(ms.int64)
        expert_output = self._ffn(sorted_input_tensor, w1, w2, group_list,
                                  activation)
        moe_output = self.moe_token_unpermute(permuted_tokens=expert_output,
                                              sorted_indices=unsort_map,
                                              probs=topk_weights,
                                              padded_mode=False,
                                              restore_shape=None)
        return moe_output

    def run_ep(self, hidden_states, w1, w2, topk_ids, topk_weights, activation,
               global_num_experts):
        raise NotImplementedError("ep mode not implemented yet.")
