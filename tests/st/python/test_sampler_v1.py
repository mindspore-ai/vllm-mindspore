# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/tests/v1/sample/test_sampler.py
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
"""Test v1 sampler."""

# type: ignore
# isort: skip_file

from typing import Optional

import numpy as np
import pytest
import torch

import vllm_mindspore
from vllm.utils import make_tensor_with_pad
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from .utils import cleanup_subprocesses


def teardown_function():
    cleanup_subprocesses()


VOCAB_SIZE = 1024
NUM_OUTPUT_TOKENS = 20
CUDA_DEVICES = [f"cuda:{0}"]
MAX_NUM_PROMPT_TOKENS = 64


def _create_fake_logits(batch_size: int, vocab_size: int) -> torch.Tensor:
    fake_logits = torch.full((batch_size, vocab_size), 1e-2, dtype=torch.float)
    return fake_logits


def _create_penalty_tensor(batch_size: int, penalty_value: float,
                           device: torch.device) -> torch.Tensor:
    return torch.full((batch_size, ),
                      fill_value=penalty_value,
                      dtype=torch.float,
                      device=device)


def _create_prompt_tokens_tensor(
    prompt_token_ids: list[list[int]],
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    return make_tensor_with_pad(
        prompt_token_ids,
        pad=vocab_size,
        device=device,
        dtype=torch.int64,
        pin_memory=False,
    )


def _create_logit_bias(
    batch_size: int,
    vocab_size: int,
    bias_value: float,
) -> list[Optional[dict[int, float]]]:
    res: list[Optional[dict[int, float]]] = []
    for i in range(batch_size):
        logit_bias = {min(i, vocab_size - 1): bias_value}
        res.append(logit_bias)
    return res


def _create_allowed_token_ids(
    batch_size: int,
    vocab_size: int,
    num_allowed_token_ids: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    mask: Optional[torch.Tensor] = None
    for i in range(batch_size):
        if i % 2 == 1:
            continue
        if mask is None:
            mask = torch.zeros((batch_size, vocab_size),
                               dtype=torch.bool,
                               device=device)
        start = min(i, vocab_size - 1)
        end = min(i + num_allowed_token_ids, vocab_size - 1)
        mask[i, start:end] = True
    return mask


def _create_bad_words_token_ids(
        batch_size: int, vocab_size: int,
        bad_words_lengths: list[tuple[int]]) -> dict[int, list[list[int]]]:
    bad_words_token_ids = {}
    for batch_idx in range(batch_size):
        token_ids_single_batch = []
        for bad_words_length in bad_words_lengths:
            token_ids = np.random.choice(vocab_size,
                                         size=bad_words_length,
                                         replace=True).tolist()
            token_ids_single_batch.append(token_ids)
        bad_words_token_ids[batch_idx] = token_ids_single_batch
    if batch_size >= 2:
        # Test no bad_words for some batch
        no_bad_words_batch_idx = np.random.choice(batch_size)
        bad_words_token_ids.pop(no_bad_words_batch_idx, None)
    return bad_words_token_ids


def _update_output_token_ids_for_bad_words(
        metadata: SamplingMetadata, vocab_size: int) -> dict[int, list[int]]:
    bad_words_last_tokens = {}
    for batch_idx, bad_words_token_ids in metadata.bad_words_token_ids.items():
        output_token_ids = metadata.output_token_ids[batch_idx]
        bad_words_last_token: list[int] = []
        for i, bad_word_token_ids in enumerate(bad_words_token_ids):
            if len(bad_word_token_ids) == 1:
                # Single token id always affects logits
                bad_words_last_token.append(bad_word_token_ids[0])
            else:
                prefix_length = len(bad_word_token_ids) - 1
                has_bad_words = np.random.choice([True, False])
                if has_bad_words:
                    output_token_ids[-prefix_length:] = bad_word_token_ids[:-1]
                    bad_words_last_token.append(bad_word_token_ids[-1])
                    break  # Maximum one update to output_token_ids
                else:  # Make sure no accidental match to bad words
                    output_token_ids[-1] = (bad_word_token_ids[-2] +
                                            1) % vocab_size
        bad_words_last_tokens[batch_idx] = bad_words_last_token
    return bad_words_last_tokens


def _create_default_sampling_metadata(
    num_output_tokens: int,
    batch_size: int,
    vocab_size: int,
    device: torch.device,
) -> SamplingMetadata:
    output_token_ids: list[list[int]] = []
    prompt_token_ids: list[list[int]] = []
    for _ in range(batch_size):
        output_token_ids.append(
            np.random.randint(0, vocab_size, size=num_output_tokens).tolist())
        prompt_token_ids.append(
            np.random.randint(0,
                              vocab_size,
                              size=np.random.randint(
                                  1, MAX_NUM_PROMPT_TOKENS)).tolist())
    fake_sampling_metadata = SamplingMetadata(
        temperature=torch.full((batch_size, ), 0.0),
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        min_p=None,
        generators={},
        max_num_logprobs=0,
        prompt_token_ids=_create_prompt_tokens_tensor(prompt_token_ids,
                                                      vocab_size, device),
        output_token_ids=output_token_ids,
        frequency_penalties=_create_penalty_tensor(batch_size, 0.0, device),
        presence_penalties=_create_penalty_tensor(batch_size, 0.0, device),
        repetition_penalties=_create_penalty_tensor(batch_size, 1.0, device),
        no_penalties=True,
        min_tokens={},
        logit_bias=[None] * batch_size,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
    )
    return fake_sampling_metadata


def _generate_min_token_penalties_and_stop_tokens(
    num_output_tokens: int, batch_size: int, vocab_size: int,
    batch_indices_for_min_token_penalty: list[int]
) -> dict[int, tuple[int, set[int]]]:
    """
    Generates and returns a dict of minimum token penalties and
    corresponding stop token IDs (`min_tokens`, `stop_token_ids`) for each
    batch.

    If a batch index is included in `batch_indices_for_min_token_penalty`,
    a higher `min_tokens` value is assigned (within a randomized range),
    and a random set of stop token IDs is created. Otherwise, a lower
    `min_tokens` value is assigned, and the stop token IDs set is empty.
    """
    min_tokens: dict[int, tuple[int, set[int]]] = {}
    for index in range(batch_size):
        if index in batch_indices_for_min_token_penalty:
            min_tokens[index] = (
                np.random.randint(num_output_tokens + 1,
                                  2 * num_output_tokens),
                set(
                    np.random.randint(0, vocab_size - 1)
                    for _ in range(np.random.randint(0, vocab_size))))
        else:
            min_tokens[index] = (np.random.randint(0,
                                                   num_output_tokens), set())
    return min_tokens


def _create_weighted_output_token_list(
        batch_size: int,
        vocab_size: int) -> tuple[list[list[int]], list[list[int]]]:
    """
    Creates an output token list where each token occurs a distinct
    number of times.

    For each batch, a random subset of token IDs is selected from the
    vocabulary. The selected tokens are then added to the output token
    list, each with a different frequency.

    Returns:
        tuple[list[list[int]], list[list[int]]]:
            - The first element is the output token list, where each sublist
              corresponds to a batch and contains tokens with weighted
              frequencies.
            - The second element is a list of distinct token IDs for each
              batch, ordered by their frequency in the corresponding output
              list.
    """
    output_token_ids: list[list[int]] = []
    sorted_token_ids_in_output: list[list[int]] = []
    for _ in range(batch_size):
        distinct_token_ids = np.random.choice(vocab_size,
                                              size=np.random.randint(1, 10),
                                              replace=False).tolist()
        sorted_token_ids_in_output.append(distinct_token_ids)
        output_token_ids_for_batch = []
        for index, token_id in enumerate(distinct_token_ids):
            output_token_ids_for_batch.extend(
                [token_id for _ in range(index + 1)])
        output_token_ids.append(output_token_ids_for_batch)
    return output_token_ids, sorted_token_ids_in_output


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_sampler_min_tokens_penalty(device: str, batch_size: int):
    """
    Tests that if the number of output tokens is less than
    SamplingParams.min_tokens then we will set the logits for
    the stop token ids to -inf.
    """
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)
    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device))
    batch_indices_for_min_token_penalty = np.random.randint(
        0, batch_size - 1, size=np.random.randint(0, batch_size)).tolist()
    min_tokens = _generate_min_token_penalties_and_stop_tokens(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE,
        batch_indices_for_min_token_penalty)
    sampling_metadata.min_tokens = min_tokens
    sampler = Sampler()
    logits = sampler.apply_penalties(fake_logits, sampling_metadata)
    logits = logits.cpu()
    for batch_idx in range(batch_size):
        for token_id in range(VOCAB_SIZE):
            _, stop_token_ids = min_tokens.get(batch_idx, (0, set()))
            if token_id in stop_token_ids:
                assert logits[batch_idx][token_id] == -float("inf")
            else:
                assert logits[batch_idx][token_id] != -float("inf")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("presence_penalty", [-2.0, 2.0])
def test_sampler_presence_penalty(device: str, batch_size: int,
                                  presence_penalty: float):
    """
    Test to verify that if presence penalty is enabled then tokens
    are penalized as per their presence in the existing output.
    """
    # Create fake logits where each token is assigned the same
    # logit value.
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)
    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device))
    output_token_ids = sampling_metadata.output_token_ids
    sampling_metadata.presence_penalties = _create_penalty_tensor(
        batch_size, presence_penalty, torch.device(device))
    sampling_metadata.no_penalties = False
    sampler = Sampler()
    logits = sampler.apply_penalties(fake_logits, sampling_metadata)
    logits = logits.cpu()
    for batch_idx in range(batch_size):
        # Since all tokens initially have the same logits, the non-penalized
        # token ID will be the one with the highest logit value, while the
        # penalized token ID will be the one with the lowest logit value.
        non_penalized_token_id = logits[batch_idx].argmax().item()
        penalized_token_id = logits[batch_idx].argmin().item()
        if presence_penalty > 0:
            # If `presence_penalty` is set to a value greater than 0, it
            # indicates a preference for new tokens over those already
            # present in the output.
            # Verify that the penalized token ID exists in the output, while the
            # non-penalized token ID does not.
            assert penalized_token_id in output_token_ids[batch_idx]
            assert non_penalized_token_id not in output_token_ids[batch_idx]
        elif presence_penalty < 0:
            # If `presence_penalty` is set to a value less than 0, it indicates
            # a preference for existing tokens over new ones. Verify that the
            # non-penalized token ID exists in the output, while the penalized
            # token ID does not.
            assert non_penalized_token_id in output_token_ids[batch_idx]
            assert penalized_token_id not in output_token_ids[batch_idx]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("frequency_penalty", [-2.0, 2.0])
def test_sampler_frequency_penalty(device: str, batch_size: int,
                                   frequency_penalty: float):
    """
    Test to verify that if frequency penalty is enabled then tokens are
    penalized as per their frequency of occurrence.
    """
    # Create fake logits where each token is assigned the same
    # logit value.
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)
    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device))
    sampling_metadata.frequency_penalties = _create_penalty_tensor(
        batch_size, frequency_penalty, torch.device(device))
    output_token_ids, sorted_token_ids_in_output = \
        _create_weighted_output_token_list(
            batch_size,
            VOCAB_SIZE,
        )
    sampling_metadata.output_token_ids = output_token_ids
    sampling_metadata.no_penalties = False
    sampler = Sampler()
    logits = sampler.apply_penalties(fake_logits, sampling_metadata)
    logits = logits.cpu()
    for batch_idx in range(batch_size):
        non_penalized_token_id = logits[batch_idx].argmax().item()
        penalized_token_id = logits[batch_idx].argmin().item()
        distinct_sorted_token_ids_in_output = sorted_token_ids_in_output[
            batch_idx]
        most_frequent_token_id = distinct_sorted_token_ids_in_output[
            len(distinct_sorted_token_ids_in_output) - 1]
        if frequency_penalty > 0:
            # If `frequency_penalty` is set to > 0, it indicates
            # a preference for new tokens over existing ones. Verify that the
            # non-penalized token ID is not present in the output, while the
            # most penalized token is the one that occurs most frequently in
            # the output.
            assert (non_penalized_token_id
                    not in distinct_sorted_token_ids_in_output)
            assert penalized_token_id == most_frequent_token_id
        elif frequency_penalty < 0:
            # If `frequency_penalty` is set to < 0, it indicates
            # a preference for existing tokens over new ones. Verify that the
            # non-penalized token ID is the one that occurs most frequently
            # in the output, while the penalized token ID is one that has not
            # yet appeared.
            assert non_penalized_token_id == most_frequent_token_id
            assert penalized_token_id not in distinct_sorted_token_ids_in_output


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("repetition_penalty", [0.1, 1.9])
def test_sampler_repetition_penalty(device: str, batch_size: int,
                                    repetition_penalty: float):
    """
    Test to verify that when the repetition penalty is enabled, tokens
    are penalized based on their presence in the prompt or the existing
    output.
    """
    # Create fake logits where each token is assigned the same
    # logit value.
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)
    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device))
    sampling_metadata.repetition_penalties = _create_penalty_tensor(
        batch_size, repetition_penalty, torch.device(device))
    sampling_metadata.no_penalties = False
    sampler = Sampler()
    logits = sampler.apply_penalties(fake_logits, sampling_metadata)
    logits = logits.cpu()
    for batch_idx in range(batch_size):
        non_penalized_token_id = logits[batch_idx].argmax().item()
        penalized_token_id = logits[batch_idx].argmin().item()
        prompt_tokens = sampling_metadata.prompt_token_ids[
            batch_idx][:].tolist()
        output_tokens = sampling_metadata.output_token_ids[batch_idx]
        if repetition_penalty > 1.0:
            # If `repetition_penalty` > 1.0, verify that the non-penalized
            # token ID has not been seen before, while the penalized token ID
            # exists either in the prompt or the output.
            assert (non_penalized_token_id not in prompt_tokens
                    and non_penalized_token_id not in output_tokens)
            assert (penalized_token_id in prompt_tokens
                    or penalized_token_id in output_tokens)
        elif repetition_penalty < 1.0:
            # If `repetition_penalty` < 1.0, verify that the penalized
            # token ID has not been seen before, while the non-penalized
            # token ID exists either in the prompt or the output.
            assert (penalized_token_id not in prompt_tokens
                    and penalized_token_id not in output_tokens)
            assert (non_penalized_token_id in prompt_tokens
                    or non_penalized_token_id in output_tokens)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("min_p", [0.0, 0.1])
def test_sampler_min_p(device: str, batch_size: int, min_p: float):
    """
    Tests that when min_p is applied, tokens with probability below 
    min_p * max_prob are masked with -inf.
    """
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)

    # Create one dominant token per batch
    for i in range(batch_size):
        fake_logits[i, 0] = 10.0  # High logit for first token
        fake_logits[i, 1:] = 1e-2  # Others remain low

    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device))

    # Configure min_p parameters
    sampling_metadata.min_p = torch.full((batch_size, ), min_p, device=device)

    sampler = Sampler()
    logits = sampler.apply_min_p(fake_logits, sampling_metadata.min_p)
    logits = logits.cpu()

    for batch_idx in range(batch_size):
        for token_id in range(VOCAB_SIZE):
            if token_id == 0:
                # Dominant token should always be unmasked
                assert logits[batch_idx][token_id] != -float("inf")
            else:
                if min_p > 0.0:
                    # Non-dominant tokens should be masked when min_p > 0
                    assert logits[batch_idx][token_id] == -float("inf")
                else:
                    # No masking when min_p is 0
                    assert logits[batch_idx][token_id] != -float("inf")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("bias_value", [-0.1, 1.2])
def test_sampler_logit_bias(device: str, batch_size: int, bias_value: float):
    """
    Test to verify that when the repetition penalty is enabled, tokens
    are penalized based on their presence in the prompt or the existing
    output.
    """
    # Create fake logits where each token is assigned the same
    # logit value.
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)
    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device))
    sampling_metadata.logit_bias = _create_logit_bias(
        batch_size=batch_size,
        vocab_size=VOCAB_SIZE,
        bias_value=bias_value,
    )
    sampler = Sampler()
    logits = sampler.apply_logits_bias(fake_logits, sampling_metadata)
    logits = logits.cpu()
    for batch_idx in range(batch_size):
        logits_for_req = logits[batch_idx]
        biased_index = min(batch_idx, VOCAB_SIZE - 1)
        for token_id in range(VOCAB_SIZE):
            if biased_index == token_id:
                assert logits_for_req[token_id].item() == pytest.approx(
                    bias_value + 1e-2)
            else:
                assert logits_for_req[token_id].item() == pytest.approx(1e-2)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_allowed_token_ids", [0, 1, 2])
def test_sampler_allowed_token_ids(device: str, batch_size: int,
                                   num_allowed_token_ids: int):
    """
    Test to verify that when the repetition penalty is enabled, tokens
    are penalized based on their presence in the prompt or the existing
    output.
    """
    # Create fake logits where each token is assigned the same
    # logit value.
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)
    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device))
    mask = _create_allowed_token_ids(
        batch_size=batch_size,
        vocab_size=VOCAB_SIZE,
        num_allowed_token_ids=num_allowed_token_ids,
        device=device,
    )
    sampling_metadata.allowed_token_ids_mask = mask
    sampler = Sampler()
    logits = sampler.apply_allowed_token_ids(fake_logits, sampling_metadata)
    logits = logits.cpu()
    for batch_idx in range(batch_size):
        logits_for_req = logits[batch_idx]
        if batch_idx % 2 == 1:
            assert torch.all(logits_for_req != -float("inf"))
            continue
        for token_id in range(VOCAB_SIZE):
            start = min(batch_idx, VOCAB_SIZE - 1)
            end = min(batch_idx + num_allowed_token_ids, VOCAB_SIZE - 1)
            if token_id >= start and token_id < end:
                assert logits_for_req[token_id] == -float(
                    "inf"), f"{batch_idx}, {token_id}"
            else:
                assert logits_for_req[token_id] != -float("inf")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("bad_words_lengths", [(1, ), (1, 3), (2, 2)])
def test_sampler_bad_words(device: str, batch_size: int,
                           bad_words_lengths: list[tuple[int]]):
    """
    Test to verify that when the bad words restriction is present, tokens
    are penalized based on their match with the bad words.
    """
    # Create fake logits where each token is assigned the same
    # logit value.
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)
    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device))
    sampling_metadata.bad_words_token_ids = _create_bad_words_token_ids(
        batch_size, VOCAB_SIZE, bad_words_lengths)
    bad_words_last_tokens = _update_output_token_ids_for_bad_words(
        sampling_metadata, VOCAB_SIZE)
    sampler = Sampler()
    logits = sampler.apply_bad_words(fake_logits, sampling_metadata)
    logits = logits.cpu()
    for batch_idx in range(batch_size):
        logits_for_req = logits[batch_idx]
        for token_id in range(VOCAB_SIZE):
            if (batch_idx in bad_words_last_tokens
                    and token_id in bad_words_last_tokens[batch_idx]):
                assert logits_for_req[token_id] == -float("inf")
            else:
                assert logits_for_req[token_id] != -float("inf")
