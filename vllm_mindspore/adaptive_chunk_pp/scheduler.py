# Copyright 2025 Huawei Technologies Co., Ltd
#
# Functions is mainly Adapted from https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/core/scheduler.py
# Copyright 2025 The vLLM team.
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

import os

from typing import Optional, Tuple

from vllm.core.scheduler import SchedulingBudget, Scheduler, PartialPrefillMetadata
from vllm.sequence import SequenceGroup, SequenceStatus

def patched_can_schedule(self, *, num_new_tokens: int, num_new_seqs: int):
    # We allow num_new_tokens to be 0 when the entire sequence has
    # been cached.
    assert num_new_tokens >= 0
    assert num_new_seqs != 0
    
    # vllm-mindspore begine: 
    # Add the constraint: reminding_num_prefill_seqs > 0
    return (self.num_batched_tokens + num_new_tokens <= self.token_budget
            and self.num_curr_seqs + num_new_seqs <= self.max_num_seqs
            and self.reminding_num_prefill_seqs > 0)
    # vllm-mindspore end.
    
def patched_get_num_new_uncached_and_cached_tokens(
    self,
    seq_group: SequenceGroup,
    status: SequenceStatus,
    enable_chunking: bool,
    budget: SchedulingBudget,
    partial_prefill_metadata: Optional[PartialPrefillMetadata] = None,
) -> Tuple[int, int]:
    """
    Returns the number of new uncached and cached tokens to schedule for a
    given sequence group that's in a given `status`.

    The API could chunk the number of tokens to compute based on `budget`
    if `enable_chunking` is True. If a sequence group has multiple
    sequences (e.g., running beam search), it means it is in decoding
    phase, so chunking doesn't happen.

    Returns (0, 0) if the new token cannot be computed due to token budget.

    The cached tokens's blocks are already computed, and the attention
    backend will reuse the cached blocks rather than recomputing them. So
    the scheduler could schedule these cached tokens "for free".

    Args:
        seq_group: The sequence group to get the number of new tokens to
            schedule.
        status: The status of the sequences to get the number of new tokens
            to schedule.
        enable_chunking: Whether to chunk the number of tokens to compute.
        budget: The budget to chunk the number of tokens to compute.
        partial_prefill_metadata: information about the partial prefills
            that are currently running


    Returns:
        A tuple of two ints. The first int is the number of new uncached
        tokens to schedule. The second int is the number of cached tokens.
        If no more new tokens can be scheduled, returns (0, 0).
    """
    num_cached_new_tokens = 0
    num_uncached_new_tokens = 0

    seqs = seq_group.get_seqs(status=status)
    # Compute the number of new uncached and cached tokens for
    # each sequence.
    for seq in seqs:
        if not seq.is_prefill():
            # Decode sequences should always just have 1 uncached token
            # TODO(rickyx): Actually is this still correct for multi-step?
            num_uncached_new_tokens += 1
            continue

        num_computed_tokens_seq = seq.get_num_computed_tokens()
        all_num_new_tokens_seq = seq.get_len() - num_computed_tokens_seq
        if not self.cache_config.enable_prefix_caching:
            # If prefix caching is not enabled, all new tokens are uncached.
            num_uncached_new_tokens += all_num_new_tokens_seq
            continue

        # NOTE: the cache token might be currently in a block that's in an
        # evictor meaning that it's not yet allocated. However, we don't
        # exclude such tokens in the cache count because it will be
        # guaranteed to be allocated later if the sequence can be allocated.
        num_cached_tokens_seq = self.block_manager.get_num_cached_tokens(
            seq)

        # Sanity check.
        if num_cached_tokens_seq < num_computed_tokens_seq:
            # This should only happen with chunked prefill, and
            # the seq is still in prefill. The `num_cached_tokens_seq`
            # is the value we calculated on scheduling the first prefill.
            # For subsequent continuous prefill steps, we cached the
            # number of cache tokens for the sequence so the cached token
            # count could be less than the number of computed tokens.
            # See comments on `ComputedBlocksTracker` for more details.
            assert (
                seq.is_prefill() and seq.status == SequenceStatus.RUNNING
                and self.scheduler_config.chunked_prefill_enabled
            ), ("Number of cached tokens should not be less than the "
                "number of computed tokens for a sequence that's still "
                f"in prefill. But there are {num_cached_tokens_seq} cached "
                f"tokens and {num_computed_tokens_seq} computed tokens "
                f"for sequence {seq.seq_id}.")

        num_cached_new_tokens_seq = max(
            0, num_cached_tokens_seq - num_computed_tokens_seq)
        num_uncached_new_tokens_seq = (all_num_new_tokens_seq -
                                        num_cached_new_tokens_seq)

        num_uncached_new_tokens += num_uncached_new_tokens_seq
        num_cached_new_tokens += num_cached_new_tokens_seq

    if num_uncached_new_tokens == 0 and num_cached_new_tokens > 0:
        # For a fully cached hit sequence, we actually need to recompute the
        # last token. So we need at least 1 uncached token to schedule.
        # See ModelRunner._compute_for_prefix_cache_hit for more details.
        num_uncached_new_tokens = 1
        num_cached_new_tokens -= 1
        
    # vllm-mindspore begin:
    # Replace the original token allocation based on max-num-batched-tokens with the pre-assigned chunk-size.
    if self.enable_adaptive_chunked_prefill and seq_group.chunk_index < len(seq_group.chunk_sizes):
        if budget.reminding_num_prefill_seqs > 0:
            num_uncached_new_tokens = seq_group.get_next_chunk_size()
        else:
            num_uncached_new_tokens = 0
    # vllm-mindspore end.
    
    return num_uncached_new_tokens, num_cached_new_tokens

def add_num_batched_tokens_sub_rmd_prefill(self,
                            req_id: str,
                            num_batched_tokens: int,
                            num_cached_tokens: int = 0):
    if req_id in self._request_ids_num_batched_tokens:
        return
    assert num_cached_tokens >= 0
    assert num_batched_tokens >= 0

    self._request_ids_num_batched_tokens.add(req_id)
    self._num_batched_tokens += num_batched_tokens
    self._num_cached_tokens += num_cached_tokens
    
    # vllm-mindspore begin:
    # After scheduling the current chunk, decrement: reminding_num_prefill_seqs -= 1
    self.reminding_num_prefill_seqs -= 1
    # vllm-mindspore end.

def apply_scheduler_patch():
    if 'ADAPTIVE_CHUNK' in os.environ and os.environ['ADAPTIVE_CHUNK'] == '1':
        SchedulingBudget.reminding_num_prefill_seqs = 1
        SchedulingBudget.can_schedule = patched_can_schedule
        SchedulingBudget.add_num_batched_tokens = add_num_batched_tokens_sub_rmd_prefill
        Scheduler.enable_adaptive_chunked_prefill = True
        Scheduler._get_num_new_uncached_and_cached_tokens = patched_get_num_new_uncached_and_cached_tokens
