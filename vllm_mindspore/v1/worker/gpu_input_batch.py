# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/v1/worker/gpu_input_batch.py
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

from typing import Optional, cast

import mindspore as ms
import numpy as np
import torch
from mindspore import Tensor
from vllm.v1.sample.logits_processor import (BatchUpdateBuilder,
                                             LogitsProcessors)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.block_table import MultiGroupBlockTable

from vllm_mindspore.model_executor.models.utils import convert_pin
from vllm_mindspore.v1.utils import _copy_slice_from_np, copy_slice

_SAMPLING_EPS = 1e-5


def _make_sampling_metadata(self) -> SamplingMetadata:
    num_reqs = self.num_reqs
    if not self.all_greedy:
        temperature = _copy_slice_from_np(  # type: ignore[func-returns-value]
            self.temperature_cpu, self.temperature, num_reqs)
        temperature = temperature[:num_reqs]
    else:
        temperature = None
    if not self.no_top_p:
        _copy_slice_from_np(self.top_p_cpu, self.top_p, num_reqs)
    if not self.no_top_k:
        _copy_slice_from_np(self.top_k_cpu, self.top_k, num_reqs)

    if not self.no_penalties:
        # Since syncing these tensors is expensive only copy them
        # if necessary i.e. if there are requests which require
        # penalties to be applied during sampling.
        _copy_slice_from_np(self.frequency_penalties_cpu,
                            self.frequency_penalties, num_reqs)
        _copy_slice_from_np(self.presence_penalties_cpu,
                            self.presence_penalties, num_reqs)
        _copy_slice_from_np(self.repetition_penalties_cpu,
                            self.repetition_penalties, num_reqs)

        # The prompt tokens are used only for applying penalties during
        # the sampling process. Hence copy these tensors only when
        # there are requests which need penalties to be applied.
        prompt_token_ids = self._make_prompt_token_ids_tensor()
    else:
        prompt_token_ids = None

    allowed_token_ids_mask: Optional[Tensor] = None
    if not self.no_allowed_token_ids:
        assert self.allowed_token_ids_mask is not None
        copy_slice(self.allowed_token_ids_mask_cpu_tensor,
                   self.allowed_token_ids_mask,
                   num_reqs,
                   return_tensor=False)
        allowed_token_ids_mask = self.allowed_token_ids_mask[:num_reqs]

    return SamplingMetadata(
        temperature=temperature,
        all_greedy=self.all_greedy,
        all_random=self.all_random,
        top_p=None if self.no_top_p else self.top_p[:num_reqs],
        top_k=None if self.no_top_k else self.top_k[:num_reqs],
        generators=self.generators,
        max_num_logprobs=self.max_num_logprobs,
        prompt_token_ids=prompt_token_ids,
        frequency_penalties=self.frequency_penalties[:num_reqs],
        presence_penalties=self.presence_penalties[:num_reqs],
        repetition_penalties=self.repetition_penalties[:num_reqs],
        output_token_ids=cast(list[list[int]], self.req_output_token_ids),
        no_penalties=self.no_penalties,
        allowed_token_ids_mask=allowed_token_ids_mask,
        bad_words_token_ids=self.bad_words_token_ids,
        logitsprocs=self.logitsprocs,
    )


def _make_prompt_token_ids_tensor(self) -> Tensor:
    max_prompt_len = self.num_prompt_tokens[:self.num_reqs].max()
    prompt_token_ids = np.empty((self.num_reqs, max_prompt_len),
                                dtype=np.int64)
    prompt_token_ids[:] = self.token_ids_cpu[:self.num_reqs, :max_prompt_len]
    for i in range(self.num_reqs):
        prompt_token_ids[i, self.num_prompt_tokens[i]:] = self.vocab_size
    prompt_token_ids_cpu_tensor = ms.from_numpy(prompt_token_ids)
    return prompt_token_ids_cpu_tensor


def input_batch_init(
    self,
    max_num_reqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    device,
    pin_memory: bool,
    vocab_size: int,
    block_sizes: list[int],  # The block_size of each kv cache group
    logitsprocs=None,
    is_spec_decode: bool = False,
    is_pooling_model: bool = False,
    num_speculative_tokens: int = 0,
):
    self.is_pooling_model = is_pooling_model
    self.is_spec_decode = is_spec_decode
    self.max_num_reqs = max_num_reqs
    self.max_model_len = max_model_len
    self.max_num_batched_tokens = max_num_batched_tokens
    self.device = device
    self.pin_memory = pin_memory
    self.vocab_size = vocab_size

    self._req_ids = []
    self.req_id_to_index = {}

    # TODO(woosuk): This buffer could be too large if max_model_len is big.
    # Find a way to reduce the CPU memory usage.
    # This buffer is not directly transferred to the GPU, so it does not
    # need to be pinned.
    self.token_ids_cpu_tensor = torch.zeros(
        (max_num_reqs, max_model_len),
        device="cpu",
        dtype=torch.int32,
        pin_memory=False,
    )
    self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()
    # vllm-mindspore begin
    self.is_token_ids = torch.zeros((max_num_reqs, max_model_len),
                                    device="cpu",
                                    dtype=bool,
                                    pin_memory=False).numpy()
    # vllm-mindspore end
    # Store prompt embeddings per request to avoid OOM from large upfront
    # allocation if max_model_len is big.
    # Maps req_index -> tensor of shape (num_prompt_tokens, hidden_size)
    self.req_prompt_embeds = {}
    self.num_tokens = np.zeros(max_num_reqs, dtype=np.int32)
    self.num_tokens_no_spec = np.zeros(max_num_reqs, dtype=np.int32)
    self.num_prompt_tokens = np.zeros(max_num_reqs, dtype=np.int32)
    self.num_computed_tokens_cpu_tensor = torch.zeros(
        (max_num_reqs, ),
        device="cpu",
        dtype=torch.int32,
        pin_memory=pin_memory,
    )
    self.num_computed_tokens_cpu_tensor = convert_pin(
        self.num_computed_tokens_cpu_tensor)
    self.num_computed_tokens_cpu = \
        self.num_computed_tokens_cpu_tensor.numpy()

    # Block table.
    self.block_table = MultiGroupBlockTable(
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        pin_memory=pin_memory,
        device=device,
        block_sizes=block_sizes,
        num_speculative_tokens=num_speculative_tokens,
    )

    # Sampling-related.
    self.temperature = torch.empty((max_num_reqs, ),
                                   dtype=torch.float32,
                                   device=device)
    self.temperature_cpu_tensor = torch.empty((max_num_reqs, ),
                                              dtype=torch.float32,
                                              device="cpu",
                                              pin_memory=pin_memory)
    self.temperature_cpu_tensor = convert_pin(self.temperature_cpu_tensor)
    self.temperature_cpu = self.temperature_cpu_tensor.numpy()
    self.greedy_reqs = set()
    self.random_reqs = set()

    self.top_p = torch.empty((max_num_reqs, ),
                             dtype=torch.float32,
                             device=device)
    self.top_p_cpu_tensor = torch.empty((max_num_reqs, ),
                                        dtype=torch.float32,
                                        device="cpu",
                                        pin_memory=pin_memory)
    self.top_p_cpu_tensor = convert_pin(self.top_p_cpu_tensor)
    self.top_p_cpu = self.top_p_cpu_tensor.numpy()
    self.top_p_reqs = set()

    self.top_k = torch.empty((max_num_reqs, ),
                             dtype=torch.int32,
                             device=device)
    self.top_k_cpu_tensor = torch.empty((max_num_reqs, ),
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=pin_memory)
    self.top_k_cpu_tensor = convert_pin(self.top_k_cpu_tensor)
    self.top_k_cpu = self.top_k_cpu_tensor.numpy()
    self.top_k_reqs = set()

    # IDs of requests which do not support spec decoding
    self.spec_decode_unsupported_reqs = set()

    # Frequency penalty related data structures
    self.frequency_penalties = torch.empty((max_num_reqs, ),
                                           dtype=torch.float,
                                           device=device)
    self.frequency_penalties_cpu_tensor = torch.empty((max_num_reqs, ),
                                                      dtype=torch.float,
                                                      device="cpu",
                                                      pin_memory=pin_memory)
    self.frequency_penalties_cpu_tensor = convert_pin(
        self.frequency_penalties_cpu_tensor)
    self.frequency_penalties_cpu = \
        self.frequency_penalties_cpu_tensor.numpy()
    self.frequency_penalties_reqs = set()

    # Presence penalty related data structures
    self.presence_penalties = torch.empty((max_num_reqs, ),
                                          dtype=torch.float,
                                          device=device)
    self.presence_penalties_cpu_tensor = torch.empty((max_num_reqs, ),
                                                     dtype=torch.float,
                                                     device="cpu",
                                                     pin_memory=pin_memory)
    self.presence_penalties_cpu_tensor = convert_pin(
        self.presence_penalties_cpu_tensor)
    self.presence_penalties_cpu = self.presence_penalties_cpu_tensor.numpy()
    self.presence_penalties_reqs = set()

    # Repetition penalty related data structures
    self.repetition_penalties = torch.empty((max_num_reqs, ),
                                            dtype=torch.float,
                                            device=device)
    self.repetition_penalties_cpu_tensor = torch.empty((max_num_reqs, ),
                                                       dtype=torch.float,
                                                       device="cpu",
                                                       pin_memory=pin_memory)
    self.repetition_penalties_cpu_tensor = convert_pin(
        self.repetition_penalties_cpu_tensor)
    self.repetition_penalties_cpu = \
        self.repetition_penalties_cpu_tensor.numpy()
    self.repetition_penalties_reqs = set()

    # Speculative decoding
    # vllm-mindspore begin
    self.num_accepted_tokens_cpu_tensor = torch.ones(
        (max_num_reqs, ),
        dtype=torch.int64,
        device="cpu",
    )
    self.num_accepted_tokens_cpu_tensor = convert_pin(
        self.num_accepted_tokens_cpu_tensor)
    # vllm-mindspore end
    self.num_accepted_tokens_cpu = \
        self.num_accepted_tokens_cpu_tensor.numpy()

    # lora related
    self.request_lora_mapping = np.zeros((self.max_num_reqs, ), dtype=np.int32)
    self.lora_id_to_request_ids = {}
    self.lora_id_to_lora_request = {}

    # req_index -> generator
    # NOTE(woosuk): The indices of the requests that do not have their own
    # generator should not be included in the dictionary.
    self.generators = {}

    self.num_logprobs = {}
    # NOTE(rob): num_prompt_logprobs only includes reqs
    # that are currently in the prefill phase.
    self.num_prompt_logprobs = {}

    # To accumulate prompt logprobs tensor chunks across prefill steps.
    self.in_progress_prompt_logprobs_cpu = {}

    # Internal representation of per-step batch state changes, used for
    # reordering persistent batch and generating logitsprocs batch state
    # updates. Should reset each step.
    self.batch_update_builder = BatchUpdateBuilder()

    # TODO convert this to LogitsProcessor
    self.has_allowed_token_ids = set()
    # NOTE(lufang): In the mask tensor, if the corresponding token allowed,
    # the value is False. Since we use masked_fill_ to set -inf.
    self.allowed_token_ids_mask = None
    self.allowed_token_ids_mask_cpu_tensor = None

    # req_index -> bad_words_token_ids
    self.bad_words_token_ids = {}

    self.logits_processing_needs_token_ids = np.zeros(max_num_reqs, dtype=bool)

    self.req_output_token_ids = []

    # Store provided logitsprocs. If none are provided, initialize empty
    # data structure
    self.logitsprocs = logitsprocs or LogitsProcessors()

    # This is updated each time the batch constituents change.
    self.sampling_metadata = self._make_sampling_metadata()

    self.pooling_params = {}

    # Cached reference to the GPU tensor of previously sampled tokens
    self.prev_sampled_token_ids = None
    self.prev_sampled_token_ids_invalid_indices = None
    self.prev_req_id_to_index = None
