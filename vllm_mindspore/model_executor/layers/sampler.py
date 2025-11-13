# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/model_executor/layers/sampler.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2024-2025 The vLLM team.
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
"""A layer that samples the next tokens from the model's outputs."""

import mindspore as ms
from mindspore import mint

# (num_token_ids, num_parent_ids) per sequence group.
SampleResultType = list[tuple[list[int], list[int]]]


def _apply_top_k_top_p(
    logits: ms.Tensor,
    p: ms.Tensor,
    k: ms.Tensor,
) -> ms.Tensor:
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    # Apply top-k.
    top_k_mask = logits_sort.size(1) - k.to(ms.int64)
    # Get all the top_k values.
    top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
    top_k_mask = logits_sort < top_k_mask
    logits_sort.masked_fill_(top_k_mask, -float("inf"))

    # Apply top-p.
    probs_sort = logits_sort.softmax(-1)
    probs_sum = probs_sort.cumsum(axis=-1)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    # at least one
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    logits = mint.empty_like(logits_sort).scatter_(dim=-1,
                                                   index=logits_idx,
                                                   src=logits_sort)
    return logits


def _apply_min_p(
    logits: ms.Tensor,
    min_p: ms.Tensor,
) -> ms.Tensor:
    """
    Adapted from
    https://github.com/oobabooga/text-generation-webui/blob/3146124ec01f02c8fb1650a6517cf1b60b537aaf/modules/sampler_hijack.py#L16C17-L16C17
    """
    probs = mint.softmax(logits, dim=-1)
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    # For MindSpore: unsqueeze_ will cause error, use unsqueeze instead
    scaled_min_p = min_p.unsqueeze(dim=1) * top_probs
    tokens_to_remove = probs < scaled_min_p
    logits = logits.masked_fill_(tokens_to_remove, -float("inf"))

    return logits
