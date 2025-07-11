#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
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
# ============================================================================

import mindspore as ms


def advance_step_flashattn(num_seqs: int, num_queries: int, block_size: int,
                           input_tokens: ms.Tensor,
                           sampled_token_ids: ms.Tensor,
                           input_positions: ms.Tensor, seq_lens: ms.Tensor,
                           slot_mapping: ms.Tensor,
                           block_tables: ms.Tensor) -> None:
    """Advance a step on Ascend for existing inputs for a multi-step runner"""
    from vllm_mindspore import _C_ops as c_ops
    c_ops.advance_step_flashattn(num_seqs=num_seqs,
                                 num_queries=num_queries,
                                 block_size=block_size,
                                 input_tokens=input_tokens,
                                 sampled_token_ids=sampled_token_ids,
                                 input_positions=input_positions,
                                 seq_lens=seq_lens,
                                 slot_mapping=slot_mapping,
                                 block_tables=block_tables)
