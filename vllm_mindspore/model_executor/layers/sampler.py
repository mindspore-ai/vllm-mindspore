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

from typing import Optional
import mindspore as ms
from mindspore import mint
from vllm.model_executor.sampling_metadata import SequenceGroupToSample, SamplingMetadata
from vllm.sampling_params import SamplingType
from vllm.sequence import Logprob, PromptLogprobs, SampleLogprobs, SequenceOutput, CompletionSequenceGroupOutput

# Import native vLLM classes - msadapter handles tensor conversion automatically
from vllm.model_executor.layers.sampler import (
    SampleResultType,
    _get_next_prompt_tokens,  
)


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


def _random_sample(
    selected_seq_groups: list[SequenceGroupToSample],
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


def _greedy_sample(
    selected_seq_groups: list[SequenceGroupToSample],
    samples: ms.Tensor,
) -> SampleResultType:
    """Enhanced greedy sampling with beam search support.

    Args:
        selected_seq_groups: Selected sequence groups
        samples: [num_samples] argmax results tensor

    Returns:
        SampleResultType: [(next_token_ids, parent_ids), ...]
    """
    samples_lst = samples.asnumpy().tolist()
    sample_idx = 0
    results: SampleResultType = []

    for seq_group in selected_seq_groups:
        if not seq_group.do_sample:
            results.append(([], []))
            continue

        seq_ids = seq_group.seq_ids
        num_parent_seqs = len(seq_ids)

        # Beam search: can have multiple sequences per group
        if num_parent_seqs == 1:
            # Standard greedy sampling
            parent_ids = [0]
            next_token_ids = [samples_lst[sample_idx]]
        else:
            # Beam search: multiple parents
            parent_ids = list(range(num_parent_seqs))
            next_token_ids = samples_lst[sample_idx:sample_idx + num_parent_seqs]

        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs

    return results


def _get_ranks(x: ms.Tensor, indices: ms.Tensor) -> ms.Tensor:
    """
    Calculate ranks of chosen tokens in MindSpore with optimized memory usage.

    Args:
        x: [N, M] logprob tensor where N=num_tokens, M=vocab_size
        indices: [N] chosen token indices

    Returns:
        ms.Tensor: [N] ranks of chosen tokens (1-based)
    """
    # Memory-efficient implementation: avoid creating intermediate vals tensor
    # Use gather to get values and expand dimensions in one operation
    chosen_values = x.gather(1, indices.unsqueeze(1))  # [N, 1]
    
    # Count tokens with higher probability using broadcast comparison
    # This avoids storing the vals tensor separately
    rank_counts = (x > chosen_values).sum(1)
    
    # Return 1-based rank
    return rank_counts.add_(1)


def _get_prompt_logprob_if_needed(
    seq_group: SequenceGroupToSample,
    selected_logprobs: ms.Tensor,
    ranks: ms.Tensor,
    top_token_ids: ms.Tensor,
    top_logprobs: ms.Tensor,
    selected_logprobs_idx: int,
    top_logprob_idx: int,
):
    """Compute prompt logprobs if needed (MindSpore compatible)."""
    sampling_params = seq_group.sampling_params
    is_prompt = seq_group.is_prompt

    prompt_logprobs: Optional[PromptLogprobs] = None
    if is_prompt and sampling_params.prompt_logprobs is not None:
        prompt_logprobs = []
        num_logprobs = sampling_params.prompt_logprobs
        next_prompt_tokens = _get_next_prompt_tokens(seq_group)

        # Convert to Python lists for efficiency
        selected_logprob_items = selected_logprobs[
            selected_logprobs_idx:selected_logprobs_idx + len(next_prompt_tokens)
        ].tolist()
        rank_items = ranks[
            selected_logprobs_idx:selected_logprobs_idx + len(next_prompt_tokens)
        ].tolist()

        for idx, token_id in enumerate(next_prompt_tokens):
            # Build prompt logprobs dictionary
            prompt_logprobs_dict: dict[int, tuple[float, int]] = {
                token_id: (selected_logprob_items[idx], rank_items[idx])
            }

            # Add top-k logprobs if requested
            if num_logprobs > 0:
                top_ids = top_token_ids[top_logprob_idx, :num_logprobs].tolist()
                top_probs = top_logprobs[top_logprob_idx, :num_logprobs].tolist()
                top_ranks = range(1, num_logprobs + 1)

                prompt_logprobs_dict.update({
                    top_id: (top_prob, rank)
                    for top_id, top_prob, rank in zip(top_ids, top_probs, top_ranks)
                })

            prompt_logprobs.append({
                token_id: Logprob(*logprob_and_rank)
                for token_id, logprob_and_rank in prompt_logprobs_dict.items()
            })
            top_logprob_idx += 1

        selected_logprobs_idx += len(next_prompt_tokens)

    return prompt_logprobs, top_logprob_idx, selected_logprobs_idx


def _get_sampled_logprob_if_needed(
    seq_group: SequenceGroupToSample,
    sample_result: tuple[list[int], list[int]],
    selected_logprobs: ms.Tensor,
    ranks: ms.Tensor,
    top_token_ids: ms.Tensor,
    top_logprobs: ms.Tensor,
    selected_logprobs_idx: int,
    top_logprob_idx: int,
):
    """Compute sample logprobs for beam search candidates."""
    seq_ids = seq_group.seq_ids
    num_logprobs = seq_group.sampling_params.logprobs
    sampled_logprobs: SampleLogprobs = []
    next_token_ids, parent_seq_ids = sample_result

    if seq_group.do_sample:
        assert len(next_token_ids) > 0

        if num_logprobs is None:
            # No detailed logprobs requested
            for next_token_id in next_token_ids:
                sampled_logprobs.append({next_token_id: Logprob(float('inf'))})
        else:
            # Detailed logprobs for beam search
            selected_logprob_items = selected_logprobs[
                selected_logprobs_idx:selected_logprobs_idx + len(next_token_ids)
            ].tolist()
            rank_items = ranks[
                selected_logprobs_idx:selected_logprobs_idx + len(next_token_ids)
            ].tolist()

            for idx, (next_token_id, parent_id) in enumerate(zip(next_token_ids, parent_seq_ids)):
                # Build sampled logprobs dictionary
                sampled_logprobs_dict = {
                    next_token_id: (selected_logprob_items[idx], rank_items[idx])
                }

                # Add top-k candidates (CRITICAL for beam search)
                if num_logprobs is not None and num_logprobs > 0:
                    top_ids = top_token_ids[
                        top_logprob_idx + parent_id, :num_logprobs
                    ].tolist()
                    top_probs = top_logprobs[
                        top_logprob_idx + parent_id, :num_logprobs
                    ].tolist()
                    top_ranks = range(1, num_logprobs + 1)

                    sampled_logprobs_dict.update({
                        top_id: (top_prob, rank)
                        for top_id, top_prob, rank in zip(top_ids, top_probs, top_ranks)
                    })

                sampled_logprobs.append({
                    token_id: Logprob(*logprob_and_rank)
                    for token_id, logprob_and_rank in sampled_logprobs_dict.items()
                })

        # Update indices for next sequence group
        selected_logprobs_idx += len(next_token_ids)
        top_logprob_idx += len(seq_ids)

    return sampled_logprobs, top_logprob_idx, selected_logprobs_idx


def get_logprobs(
    logprobs: ms.Tensor,
    sampling_metadata: SamplingMetadata,
    sample_results: SampleResultType,
) -> tuple[list[Optional[PromptLogprobs]], list[SampleLogprobs]]:
    """Calculate logprobs for beam search candidates.

    This is the CRITICAL function for beam search - it generates the
    top-k candidates that beam search uses for expansion.

    Args:
        logprobs: [num_tokens, vocab_size] model log probabilities
        sampling_metadata: Sampling configuration and metadata
        sample_results: Sampling results from _sample()

    Returns:
        tuple: (prompt_logprobs, sample_logprobs) for each sequence group
    """
    # Note: No conversion needed - msadapter handles all tensor conversion

    # Collect query indices and next token IDs
    query_indices: list[int] = []
    next_token_ids: list[int] = []
    largest_num_logprobs = -1

    # Process each sequence group for logprob collection
    for (seq_group, sample_result) in zip(sampling_metadata.seq_groups, sample_results):
        sampling_params = seq_group.sampling_params

        # Handle prompt logprobs
        if (seq_group.is_prompt and sampling_params.prompt_logprobs is not None):
            largest_num_logprobs = max(largest_num_logprobs, sampling_params.prompt_logprobs)
            next_prompt_tokens = _get_next_prompt_tokens(seq_group)
            query_indices.extend(seq_group.prompt_logprob_indices)
            next_token_ids.extend(next_prompt_tokens)

        # Handle sample logprobs (KEY for beam search)
        if seq_group.do_sample:
            token_ids, parent_seq_ids = sample_result
            query_idx = seq_group.sample_indices[0]

            # Extend query indices for all parent sequences
            query_indices.extend([query_idx + parent_id for parent_id in parent_seq_ids])
            next_token_ids.extend(token_ids)

            # Update largest logprobs count (beam search uses this)
            if sampling_params.logprobs is not None:
                largest_num_logprobs = max(largest_num_logprobs, sampling_params.logprobs)

    if len(query_indices) == 0:
        # No logprobs needed
        empty_sampled_logprob: SampleLogprobs = []
        empty_prompt_logprob: Optional[PromptLogprobs] = None
        num_seq_groups = len(sampling_metadata.seq_groups)
        return [empty_prompt_logprob] * num_seq_groups, [empty_sampled_logprob] * num_seq_groups

    selected_logprobs, ranks = None, None
    top_logprobs, top_token_ids = None, None

    # Calculate logprobs if needed
    if largest_num_logprobs >= 0:
        # Convert to MindSpore tensors
        query_indices_gpu = ms.Tensor(query_indices, dtype=ms.int64)
        next_token_ids_gpu = ms.Tensor(next_token_ids, dtype=ms.int64)

        # Extract selected token logprobs
        selected_logprobs = logprobs[query_indices_gpu, next_token_ids_gpu]

        # Calculate ranks using MindSpore operations
        ranks = _get_ranks(logprobs[query_indices_gpu], next_token_ids_gpu)

        # Get top-k logprobs for beam search candidates
        if largest_num_logprobs > 0:
            top_logprobs, top_token_ids = mint.topk(logprobs, largest_num_logprobs, dim=-1)
            top_logprobs = top_logprobs.asnumpy()
            top_token_ids = top_token_ids.asnumpy()

        # Transfer to CPU
        selected_logprobs = selected_logprobs.asnumpy()
        ranks = ranks.asnumpy()

    # Build final logprobs results
    prompt_logprobs_per_seq_group: list[Optional[PromptLogprobs]] = []
    sample_logprobs_per_seq_group: list[SampleLogprobs] = []
    top_logprob_idx = 0
    selected_logprobs_idx = 0

    for seq_group, sample_result in zip(sampling_metadata.seq_groups, sample_results):
        # Process prompt logprobs
        (prompt_logprobs, top_logprob_idx, selected_logprobs_idx) = _get_prompt_logprob_if_needed(
            seq_group, selected_logprobs, ranks, top_token_ids, top_logprobs,
            selected_logprobs_idx, top_logprob_idx
        )
        prompt_logprobs_per_seq_group.append(prompt_logprobs)

        # Process sample logprobs (CRITICAL for beam search)
        (sampled_logprobs, top_logprob_idx, selected_logprobs_idx) = _get_sampled_logprob_if_needed(
            seq_group, sample_result, selected_logprobs, ranks, top_token_ids,
            top_logprobs, selected_logprobs_idx, top_logprob_idx
        )
        sample_logprobs_per_seq_group.append(sampled_logprobs)

    return prompt_logprobs_per_seq_group, sample_logprobs_per_seq_group