# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/model_executor/layers/logits_processor.py
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
"""A layer that compute logits from hidden_stats."""
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import vllm.envs as envs
from mindspore import Tensor, jit, mint, nn
from vllm.config import current_platform, get_current_vllm_config
from vllm.distributed import tensor_model_parallel_gather
from vllm.model_executor.sampling_metadata import SamplingMetadata

from vllm_mindspore.distributed.communication_op import (
    AllGatherFromModelParallelRegion)
from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)

_logits_processor_threadpool: Optional[ThreadPoolExecutor] = None
if envs.VLLM_LOGITS_PROCESSOR_THREADS is not None:
    _logits_processor_threadpool = ThreadPoolExecutor(
        envs.VLLM_LOGITS_PROCESSOR_THREADS)


class LogitsProcessor(nn.Cell):
    """Process logits and apply logits processors from sampling metadata.

    This layer does the following:
    1. Gather logits from model hidden_states.
    2. Scale logits if needed.
    3. Apply logits processors (if any).
    """

    def __init__(
        self,
        vocab_size: int,
        org_vocab_size: Optional[int] = None,
        scale: float = 1.0,
        logits_as_input: bool = False,
        soft_cap: Optional[float] = None,
    ) -> None:
        """
        Args:
            scale: A scaling factor to apply to the logits.
        """
        super().__init__()
        vllm_config = get_current_vllm_config()
        self.vllm_config = vllm_config
        self.is_graph_mode = bool(not vllm_config.model_config.enforce_eager)
        self.scale = scale
        self.vocab_size = vocab_size
        # Whether the input is logits (default is hidden states).
        self.logits_as_input = logits_as_input
        # original vocabulary size (without LoRA).
        self.org_vocab_size = org_vocab_size or vocab_size
        # Soft cap the logits. Used in Gemma 2.
        self.soft_cap = soft_cap
        # Whether to use gather or all-gather to gather the logits.
        self.use_all_gather = current_platform.use_all_gather()

        if self.use_all_gather:
            self.tensor_model_parallel_all_gather = \
                AllGatherFromModelParallelRegion()
        self.lm_head = None
        self.run_model = None
        self.cached_input_info = {}

    def set_dynamic_inputs(self):
        dyn_hidden_states = Tensor(shape=[None, None],
                                   dtype=self.vllm_config.model_config.dtype)

        if self.cached_input_info["indices"] is None:
            dyn_indices = None
        else:
            dyn_indices_shape = [
                None for _ in range(self.cached_input_info["indices"]["ndim"])
            ]
            dyn_indices_dtype = self.cached_input_info["indices"]["dtype"]
            dyn_indices = Tensor(shape=dyn_indices_shape,
                                 dtype=dyn_indices_dtype)

        if self.cached_input_info["bias"] is None:
            dyn_bias = None
        else:
            dyn_bias_shape = [
                None for _ in range(self.cached_input_info["bias"]["ndim"])
            ]
            dyn_bias_dtype = self.cached_input_info["bias"]["dtype"]
            dyn_bias = Tensor(shape=dyn_bias_shape, dtype=dyn_bias_dtype)

        self.set_inputs(dyn_hidden_states, dyn_indices, dyn_bias)

    def __call__(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: Tensor,
        sampling_metadata: Optional[SamplingMetadata] = None,
        embedding_bias: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        if self.lm_head is None:
            self.lm_head = lm_head
        if self.run_model is None:
            self.run_model = jit(
                function=self.construct,
                jit_level='O0') if self.is_graph_mode else self.construct
        selected_token_indices = None
        if sampling_metadata is not None:
            selected_token_indices = sampling_metadata.selected_token_indices
        dyn_indices_info = None if selected_token_indices is None else {
            "ndim": selected_token_indices.ndim,
            "dtype": selected_token_indices.dtype,
        }
        dyn_bias_info = None if embedding_bias is None else {
            "ndim": embedding_bias.ndim,
            "dtype": embedding_bias.dtype,
        }
        if self.cached_input_info != {
                "indices": dyn_indices_info,
                "bias": dyn_bias_info
        }:
            self.cached_input_info = {
                "indices": dyn_indices_info,
                "bias": dyn_bias_info,
            }
            self.set_dynamic_inputs()

        if selected_token_indices is not None and selected_token_indices.numel(
        ) <= 0:
            logits = mint.zeros((0, self.vocab_size),
                                dtype=hidden_states.dtype)
        else:
            logits = self.run_model(hidden_states, selected_token_indices,
                                    embedding_bias)

        if sampling_metadata is not None and \
                sampling_metadata.seq_groups is not None:
            logits = _apply_logits_processors(logits, sampling_metadata)

        return logits

    def construct(
        self,
        hidden_states: Tensor,
        selected_token_indices: Optional[Tensor] = None,
        embedding_bias: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        if self.logits_as_input:
            logits = hidden_states
        else:
            if selected_token_indices is not None:
                hidden_states = mint.index_select(hidden_states, 0,
                                                  selected_token_indices)

            # Get the logits for the next tokens.
            logits = self._get_logits(hidden_states, self.lm_head,
                                      embedding_bias)
        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = mint.tanh(logits)
                logits = logits * self.soft_cap

            if self.scale != 1.0:
                logits *= self.scale

            # Apply logits processors (if any).
        return logits

    def _get_logits(
        self,
        hidden_states: Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[Tensor],
    ) -> Optional[Tensor]:
        # Get the logits for the next tokens.
        logits = lm_head.quant_method.apply(lm_head,
                                            hidden_states,
                                            bias=embedding_bias)
        if self.use_all_gather:
            # Gather is not supported for some devices such as NPUs.
            logits = self.tensor_model_parallel_all_gather(logits)
        else:
            # None may be returned for rank > 0
            logits = tensor_model_parallel_gather(logits)
        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[..., :self.org_vocab_size]
        return logits

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", forg_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s


def _apply_logits_processors(
    logits: Tensor,
    sampling_metadata: SamplingMetadata,
) -> Tensor:
    found_logits_processors = False
    logits_processed = 0
    logits_row_ids_and_logits_row_futures = []
    for seq_group in sampling_metadata.seq_groups:
        seq_ids = seq_group.seq_ids
        sampling_params = seq_group.sampling_params
        logits_processors = sampling_params.logits_processors
        if logits_processors:
            found_logits_processors = True

            for seq_id, logits_row_idx in zip(seq_ids,
                                              seq_group.sample_indices):
                logits_row = logits[logits_row_idx]
                past_tokens_ids = seq_group.seq_data[seq_id].output_token_ids
                prompt_tokens_ids = seq_group.seq_data[seq_id].prompt_token_ids

            if _logits_processor_threadpool is not None:
                logits_row_ids_and_logits_row_futures.append(
                    (logits_row_idx,
                     _logits_processor_threadpool.submit(
                         _apply_logits_processors_single_seq, logits_row,
                         logits_processors, past_tokens_ids,
                         prompt_tokens_ids)))
            else:
                logits[logits_row_idx] = \
                    _apply_logits_processors_single_seq(
                        logits_row, logits_processors, past_tokens_ids,
                        prompt_tokens_ids)

        logits_processed += len(seq_group.sample_indices) + len(
            seq_group.prompt_logprob_indices)

    for logits_row_idx, future in logits_row_ids_and_logits_row_futures:
        logits[logits_row_idx] = future.result()

    if found_logits_processors:
        # verifies that no rows in logits were missed unexpectedly
        assert logits_processed == logits.shape[0]
    return logits


def _apply_logits_processors_single_seq(logits_row, logits_processors,
                                        past_tokens_ids,
                                        prompt_tokens_ids) -> Tensor:
    for logits_processor in logits_processors:
        parameters = inspect.signature(logits_processor).parameters
        if len(parameters) == 3:
            logits_row = logits_processor(prompt_tokens_ids, past_tokens_ids,
                                          logits_row)
        else:
            logits_row = logits_processor(past_tokens_ids, logits_row)
    return logits_row
