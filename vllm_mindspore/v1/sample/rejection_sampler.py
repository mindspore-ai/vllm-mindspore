# SPDX-License-Identifier: Apache-2.0

# Functions are adapted from
# https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/sample/rejection_sampler.py
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
"""replace triton ops with ms operators."""
from typing import Optional

import mindspore as ms
from mindspore import mint
from vllm.logger import init_logger
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import generate_uniform_probs

logger = init_logger(__name__)

PLACEHOLDER_TOKEN_ID = -1
GREEDY_TEMPERATURE = -1


def expand_batch_to_tokens(
    x: ms.Tensor,  # [batch_size]
    cu_num_tokens: ms.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> ms.Tensor:
    """Expand [batch_size] tensor to [num_tokens] tensor based on the number of
    tokens per batch in cu_num_tokens.

    For example, if x = [a, b, c] and cu_num_tokens = [2, 5, 6], then
    num_tokens = 6, and expanded_x = [a, a, b, b, b, c].

    Args:
        x: [batch_size] tensor to expand.
        cu_num_tokens: [batch_size] tensor containing the cumulative number of
            tokens per batch. Each element represents the total number of
            tokens up to and including that batch.
        num_tokens: Total number of tokens.
        replace_from: int = 0
            Value to be replaced if it is found in x.
        replace_to: int = 0
            Value to replace with when replace_from is found.
    Returns:
        expanded_x: [num_tokens] tensor.
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size

    tokens_per_batch = cu_num_tokens.clone()
    tokens_per_batch[1:] -= cu_num_tokens[:-1]
    expanded_x = mint.repeat_interleave(x, tokens_per_batch, dim=0)
    if replace_from != replace_to:
        expanded_x = mint.where(expanded_x == replace_from,
                                ms.tensor(replace_to, dtype=expanded_x.dtype),
                                expanded_x)

    current_len = expanded_x.shape[0]
    if current_len > num_tokens:
        expanded_x = expanded_x[:num_tokens]
    elif current_len < num_tokens:
        pad_len = num_tokens - current_len
        padding = mint.zeros(pad_len, dtype=expanded_x.dtype)
        expanded_x = mint.cat([expanded_x, padding], dim=0)

    return expanded_x


def rejection_sample(
    draft_token_ids: ms.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: ms.Tensor,
    draft_probs: Optional[ms.Tensor],
    target_probs: ms.Tensor,
    bonus_token_ids: ms.Tensor,
    sampling_metadata: SamplingMetadata,
) -> ms.Tensor:
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_probs.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_probs.shape[-1]
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert target_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_probs.shape == (num_tokens, vocab_size)

    # Create output buffer.
    output_token_ids = mint.empty((batch_size, max_spec_len + 1),
                                  dtype=ms.int32)
    output_token_ids.fill_(PLACEHOLDER_TOKEN_ID)

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        target_argmax = target_probs.argmax(dim=-1)
        _rejection_greedy_sample(output_token_ids, num_draft_tokens,
                                 cu_num_draft_tokens, draft_token_ids,
                                 target_argmax, bonus_token_ids, is_greedy)
        if sampling_metadata.all_greedy:
            return output_token_ids

    # Generate uniform probabilities for rejection sampling.
    uniform_probs = generate_uniform_probs(num_tokens, num_draft_tokens,
                                           sampling_metadata.generators,
                                           target_probs.device)

    # Sample recovered tokens for each position.
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len, num_draft_tokens, cu_num_draft_tokens, draft_token_ids,
        draft_probs, target_probs, sampling_metadata)

    # Rejection sampling for random sampling requests.
    _rejection_random_sample(output_token_ids, cu_num_draft_tokens,
                             draft_token_ids, draft_probs, target_probs,
                             bonus_token_ids, recovered_token_ids,
                             uniform_probs, is_greedy, max_spec_len,
                             vocab_size)
    return output_token_ids


def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: ms.Tensor,
    draft_token_ids: ms.Tensor,
    draft_probs: Optional[ms.Tensor],
    target_probs: ms.Tensor,
    sampling_metadata: SamplingMetadata,
) -> ms.Tensor:
    # NOTE(woosuk): Create only one distribution for each request.
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q = mint.empty((batch_size, vocab_size), dtype=ms.float32)
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

    recovered_token_ids = mint.empty_like(draft_token_ids)
    _sample_recovered_tokens(recovered_token_ids, cu_num_draft_tokens,
                             draft_token_ids, draft_probs, target_probs, q,
                             vocab_size)
    return recovered_token_ids


def _sample_recovered_tokens(
    output_token_ids,  # [num_tokens]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size]
    target_probs,  # [num_tokens, vocab_size]
    q,  # [batch_size, vocab_size]
    vocab_size,
):
    batch_size = cu_num_draft_tokens.numel()

    for req_idx in range(batch_size):
        start_idx = 0 if req_idx == 0 else \
            int(cu_num_draft_tokens[req_idx - 1])
        end_idx = int(cu_num_draft_tokens[req_idx])
        num_draft = end_idx - start_idx

        for pos in range(num_draft):
            token_idx = start_idx + pos
            if draft_probs is None:
                draft_token_id = draft_token_ids[token_idx]
                prob = target_probs[token_idx].clone()
                prob[draft_token_id] = 0.0
            else:
                draft_prob = draft_probs[token_idx]
                target_prob = target_probs[token_idx]
                prob = mint.maximum(target_prob - draft_prob,
                                    mint.zeros_like(target_prob))

            q_vec = q[req_idx]
            recovered_id = mint.argmax(prob / q_vec)
            output_token_ids[token_idx] = recovered_id
    return


def _rejection_greedy_sample(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    num_draft_tokens,  # [batch_size]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    target_argmax,  # [num_tokens]
    bonus_token_ids,  # [batch_size]
    is_greedy=None,
):
    batch_size = output_token_ids.shape[0]
    if num_draft_tokens.count(1) == len(num_draft_tokens):
        num_tokens = draft_token_ids.shape[0]
        assert batch_size == num_tokens
        output_token_ids[:, 0] = target_argmax
        output_token_ids[:, 1] = mint.where(draft_token_ids == target_argmax,
                                            bonus_token_ids.reshape(-1),
                                            output_token_ids[:, 1])
        return

    if is_greedy is None:
        is_greedy = mint.ones(batch_size, dtype=ms.bool)
    else:
        is_greedy = is_greedy.bool()

    start_idx = mint.zeros(batch_size, dtype=ms.long)
    start_idx[1:] = cu_num_draft_tokens[:-1]
    end_idx = cu_num_draft_tokens

    for req_idx in range(batch_size):
        if not is_greedy[req_idx]:
            continue
        s = start_idx[req_idx].item()
        e = end_idx[req_idx].item()
        draft_tokens = draft_token_ids[s:e]
        target_tokens = target_argmax[s:e]

        rejected = False
        for pos, (d, t) in enumerate(zip(draft_tokens, target_tokens)):
            if not rejected:
                output_token_ids[req_idx, pos] = t
            if not rejected and d != t:
                rejected = True
        if not rejected:
            output_token_ids[req_idx,
                             len(draft_tokens)] = bonus_token_ids[req_idx]
    return


def _rejection_random_sample(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, vocab_size]
    bonus_token_ids,  # [batch_size]
    recovered_token_ids,  # [num_tokens]
    uniform_probs,  # [num_tokens]
    is_greedy,  # [batch_size]
    max_spec_len,
    vocab_size,
):
    batch_size = len(cu_num_draft_tokens)

    for req_idx in range(batch_size):
        if is_greedy[req_idx]:
            continue

        start_idx = 0 if req_idx == 0 else cu_num_draft_tokens[req_idx - 1]
        end_idx = cu_num_draft_tokens[req_idx]
        num_tokens = end_idx - start_idx
        rejected = False
        for pos in range(num_tokens):
            if not rejected:
                draft_token_id = draft_token_ids[start_idx + pos]
                draft_prob_val = 1 if draft_probs is None else \
                    draft_probs[start_idx + pos, draft_token_id]
                target_prob_val = target_probs[start_idx + pos, draft_token_id]
                uniform_prob_val = uniform_probs[start_idx + pos]

                if draft_prob_val > 0 and \
                    target_prob_val / draft_prob_val >= uniform_prob_val:
                    token_id = draft_token_id
                else:
                    token_id = recovered_token_ids[start_idx + pos]
                    rejected = True
                output_token_ids[req_idx, pos] = token_id

        if not rejected:
            bonus_token_id = bonus_token_ids[req_idx]
            output_token_ids[req_idx, num_tokens] = bonus_token_id
    return
