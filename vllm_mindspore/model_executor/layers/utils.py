# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/model_executor/layers/utils.py
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
"""Utility methods for model layers."""
import mindspore as ms


def get_token_bin_counts_and_mask(
    tokens: ms.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> tuple[ms.Tensor, ms.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = ms.mint.zeros((num_seqs, vocab_size + 1), dtype=ms.int64)
    bin_counts.scatter_add_(1, tokens, ms.mint.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def get_repetition_penalties_mask(
    prompt_tokens: ms.Tensor,
    output_tokens: ms.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> ms.Tensor:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = ms.mint.zeros((num_seqs, vocab_size + 1), dtype=ms.int64)
    bin_counts.scatter_add_(1, prompt_tokens, ms.mint.ones_like(prompt_tokens))
    bin_counts.scatter_add_(1, output_tokens, ms.mint.ones_like(output_tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return mask


def apply_penalties(logits: ms.Tensor, prompt_tokens_tensor: ms.Tensor,
                    output_tokens_tensor: ms.Tensor,
                    presence_penalties: ms.Tensor,
                    frequency_penalties: ms.Tensor,
                    repetition_penalties: ms.Tensor) -> ms.Tensor:
    """
    Applies penalties out of place implement to improve performance.
    logits : The input logits tensor of shape [num_seqs, vocab_size]
    prompt_tokens_tensor: A tensor containing the prompt tokens. The prompts 
        are padded to the maximum prompt length within the batch using 
        `vocab_size` as the padding value. The value `vocab_size` is used 
        for padding because it does not correspond to any valid token ID 
        in the vocabulary.
    output_tokens_tensor: The output tokens tensor.
    presence_penalties: The presence penalties of shape (num_seqs, )
    frequency_penalties: The frequency penalties of shape (num_seqs, )
    repetition_penalties: The repetition penalties of shape (num_seqs, )
    """
    if logits.numel() <= 0:
        return logits
    num_seqs, vocab_size = logits.shape

    if repetition_penalties is not None:
        mask = get_repetition_penalties_mask(
            prompt_tokens_tensor,
            output_tokens_tensor,
            vocab_size,
            num_seqs,
        )
        # use 'broadcast_to' to replace 'tensor.repeat' to improve performance
        # when tensor shape is (num,seqs, 1), 'tensor.repeat(1, vocab_size)'
        # is equal to 'broadcast_to(tensor, (num_seqs, vocab_size))'
        repetition_penalties = ms.mint.broadcast_to(
            repetition_penalties.unsqueeze(dim=1), (num_seqs, vocab_size))

        # use out of place computation instead of inplace setitem to improve
        # performance 'tensor[tensor > 0]' will result in setitem,
        # which is slow.
        logits = ms.mint.where(mask & (logits > 0),
                               logits / repetition_penalties, logits)
        logits = ms.mint.where(mask & (logits <= 0),
                               logits * repetition_penalties, logits)
    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details\
    if frequency_penalties is not None or presence_penalties is not None:
        output_bin_counts, output_mask = get_token_bin_counts_and_mask(
            output_tokens_tensor, vocab_size, num_seqs)
        if frequency_penalties is not None:
            logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
        if presence_penalties is not None:
            logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits
