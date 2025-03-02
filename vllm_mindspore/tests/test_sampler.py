#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
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
# ============================================================================
import vllm_mindspore
import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_mindspore.model_executor.layers.sampler import Sampler
from vllm_mindspore.model_executor.sampling_metadata import SamplingMetadata
# from vllm_mindspore.model_executor.utils import set_random_seed
from vllm_mindspore.sequence import SamplingParams, SequenceData, SequenceGroupMetadata

VOCAB_SIZE = 32000
RANDOM_SEEDS = list(range(128))

class MockLogitsSampler(Sampler):

    def __init__(self, fake_logits: torch.Tensor):
        super().__init__()
        self.fake_logits = fake_logits

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

def _prepare_test(
        batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor, MockLogitsSampler]:
    input_tensor = torch.rand((batch_size, 1024), dtype=torch.float16)
    fake_logits = torch.full((batch_size, VOCAB_SIZE),
                             1e-2,
                             dtype=input_tensor.dtype)
    sampler = MockLogitsSampler(fake_logits)
    return input_tensor, fake_logits, sampler

def _do_sample(
    batch_size: int,
    input_tensor: torch.Tensor,
    sampler: MockLogitsSampler,
    sampling_params: SamplingParams,
    device: str,
):
    seq_group_metadata_list: List[SequenceGroupMetadata] = []
    seq_lens: List[int] = []
    for i in range(batch_size):
        seq_group_metadata_list.append(
            SequenceGroupMetadata(
                request_id=f"test_{i}",
                is_prompt=True,
                seq_data={0: SequenceData.from_seqs([1, 2, 3])},
                sampling_params=sampling_params,
                block_tables={0: [1]},
            ))
        seq_lens.append(seq_group_metadata_list[-1].seq_data[0].get_len())

    sampling_metadata = SamplingMetadata.prepare(
        seq_group_metadata_list,
        seq_lens,
        query_lens=seq_lens,
        device=device,
        pin_memory=False)
    return sampler(logits=input_tensor, sampling_metadata=sampling_metadata)

def test_sampler_all_greedy():
    # set_random_seed(seed)
    device='cuda'
    # torch.set_default_device(device)
    batch_size = random.randint(1, 256)
    input_tensor, fake_logits, sampler = _prepare_test(batch_size)

    sampling_params = SamplingParams(temperature=0)
    sampler_output = _do_sample(batch_size, fake_logits, sampler,
                                sampling_params, device)
    expected = mint.argmax(fake_logits, dim=-1)
    for i, sequence_output in enumerate(sampler_output):
        for nth_output in sequence_output.samples:
            assert nth_output.output_token == expected[i].item()


def test_sampler_all_random():
    # set_random_seed(seed)
    # torch.set_default_device(device)
    device='cuda'
    batch_size = random.randint(1, 256)
    _, fake_logits, sampler = _prepare_test(batch_size)

    for i in range(batch_size):
        fake_logits[i, i] = 1e2

    sampling_params = SamplingParams(
        temperature=1.0,
        n=random.randint(1, 10),
    )
    sampler_output = _do_sample(batch_size, fake_logits, sampler,
                                sampling_params, device)

    for i, sequence_output in enumerate(sampler_output):
        for nth_output in sequence_output.samples:
            assert nth_output.output_token == i



def test_sampler_repetition_penalty_mixed():
    device='cuda'
    vocab_size = 8

    def test_sampling_params(sampling_params: List[SamplingParams]):

        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        seq_lens: List[int] = []
        for i in range(2):
            seq_group_metadata_list.append(
                SequenceGroupMetadata(
                    request_id=f"test_{i}",
                    is_prompt=True,
                    seq_data={0: SequenceData.from_seqs([1, 2, 3])},
                    sampling_params=sampling_params[i],
                    block_tables={0: [1]},
                ))
            seq_lens.append(seq_group_metadata_list[-1].seq_data[0].get_len())

        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            seq_lens,
            query_lens=seq_lens,
            device=device,
            pin_memory=False)

        fake_logits = torch.full((2, vocab_size),
                                 1e-2,
                                 dtype=torch.float16)

        fake_logits[:, 5] = 1.1e-2
        fake_logits[:, 1] = 1.2e-2

        sampler = MockLogitsSampler(fake_logits)
        print(f'fake_logits is: {fake_logits}', flush = True)

        sampler_output = sampler(logits=fake_logits,
                                 sampling_metadata=sampling_metadata)

        generated_tokens = []
        for output in sampler_output:
            generated_tokens.append(output.samples[0].output_token)

        return generated_tokens

    # one configuration is greedy with repetition_penalty
    sampling_params_rep = SamplingParams(
        temperature=0.0,
        repetition_penalty=2.0,
    )

    # other configuration is sampling w/o repetition_penalty
    sampling_params_sample = SamplingParams(
        temperature=1.0,
        top_k=1,
    )

    tokens1 = test_sampling_params(
        [sampling_params_rep, sampling_params_sample])

    tokens2 = test_sampling_params(
        [sampling_params_sample, sampling_params_rep])

    assert tokens1[0] == tokens2[1]