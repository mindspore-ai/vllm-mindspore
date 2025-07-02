#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
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

from typing import List, Tuple
from math import inf

import mindspore as ms

from vllm.model_executor.sampling_metadata import (
    SequenceGroupToSample
)

# (num_token_ids, num_parent_ids) per sequence group.
SampleResultType = List[Tuple[List[int], List[int]]]

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
    # for vllm-mindspore: softmax and cumsum has performance issues
    probs_sort = logits_sort.softmax(-1)
    probs_sum = probs_sort.cumsum(axis=-1)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    # at least one
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    logits = ms.mint.empty_like(logits_sort).scatter_(dim=-1,
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
    probs = ms.mint.softmax(logits, dim=-1)
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    # For vllm-mindspore: unsqueeze_ will cause error, use unsqueeze instead
    scaled_min_p = min_p.unsqueeze(dim=1) * top_probs
    tokens_to_remove = probs < scaled_min_p
    logits = logits.masked_fill_(tokens_to_remove, -float("inf"))

    return logits

def _random_sample(
    selected_seq_groups: List[SequenceGroupToSample],
    random_samples: ms.Tensor,
) -> SampleResultType:
    """Run random sampling on a given samples.

    Args:
        selected_seq_groups: A list of sequence groups batched.
        random_samples: (num_selected_samples,) A tensor of samples. The
            length of samples could be smaller than selected_seq_groups if
            seq_group.do_sample is False.
    Returns:
        Tuple of (next_token_ids, parent_ids). The length of returned list is
        same as the length of selected_seq_groups. If the corresponding
        seq_group has do_sample=False, tuple contains ([], [])
    """
    # Find the maximum n value of the prompt phase requests.
    sample_idx = 0
    results: SampleResultType = []
    # random_samples = random_samples.cpu()
    random_samples = random_samples.asnumpy()
    for seq_group in selected_seq_groups:
        if not seq_group.do_sample:
            results.append(([], []))
            continue

        seq_ids = seq_group.seq_ids
        sampling_params = seq_group.sampling_params
        is_prompt = seq_group.is_prompt
        num_parent_seqs = len(seq_ids)
        if is_prompt:
            # Prompt phase.
            parent_ids = [0] * sampling_params.n
            next_token_ids = random_samples[
                sample_idx, :sampling_params.n].tolist()
        else:
            # Generation phase.
            parent_ids = list(range(num_parent_seqs))
            next_token_ids = random_samples[sample_idx:sample_idx +
                                            num_parent_seqs, 0].tolist()
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    return results
