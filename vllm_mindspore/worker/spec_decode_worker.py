# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/spec_decode/spec_decode_worker.py
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

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from vllm.worker.worker_base import WorkerBase
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.metrics import AsyncMetricsCollector
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeScorer, SpeculativeScores)
from vllm.sequence import (VLLM_INVALID_TOKEN_ID,
                           CompletionSequenceGroupOutput, ExecuteModelRequest,
                           HiddenStates, SequenceGroupMetadata,
                           get_all_seq_ids_and_request_ids)
from vllm.model_executor.layers.spec_decode_base_sampler import (
    SpecDecodeBaseSampler, SpecDecodeStochasticBaseSampler)

from vllm.spec_decode.util import (Timer, create_logprobs_output,
                                   create_sequence_group_output,
                                   get_all_num_logprobs,
                                   get_sampled_token_logprobs, nvtx_range,
                                   split_batch_by_proposal_len)

# MQAScore is only supported in FLASH_ATTN and eager mode.
def spec_decode_worker_init(
        self,
        proposer_worker: ProposerWorkerBase,
        scorer_worker: WorkerBase,
        spec_decode_sampler: SpecDecodeBaseSampler,
        disable_mqa_scorer: bool = False,
        disable_logprobs: bool = False,
        disable_log_stats: bool = False,
        metrics_collector: Optional[AsyncMetricsCollector] = None,
        disable_by_batch_size: Optional[int] = None,
        allow_zero_draft_token_step: Optional[bool] = True,
        enable_lm_head_weight_load: Optional[bool] = False,
        num_spec_prefill_steps: int = 1,
):
    self.proposer_worker = proposer_worker
    self.scorer_worker = scorer_worker
    scorer_runner = getattr(self.scorer_worker, "model_runner", None)
    self.generators = scorer_runner.get_generators(
    ) if scorer_runner else None
    self.disable_by_batch_size = disable_by_batch_size or float("inf")
    self.spec_decode_sampler = spec_decode_sampler
    self._allow_zero_draft_token_step = allow_zero_draft_token_step
    self._enable_lm_head_weight_load = enable_lm_head_weight_load
    self._metrics = AsyncMetricsCollector(
        self.spec_decode_sampler
    ) if metrics_collector is None else metrics_collector
    # Tracks the sequence IDs that received a bonus token ID in
    # their last forward pass. Needed only if KV cache is being
    # used for token generation such as in the case of MultiStepWorker.
    self._seq_with_bonus_token_in_last_step: Set[int] = set()
    # Tracks the currently active request ids and the sequence IDs
    # corresponding to them
    self._request_id_seq_id_mapping: Dict[str, Set[int]] = defaultdict(set)
    # Tracks if the proposer worker uses the KV cache or not.

    self.probs_dtype = self.spec_decode_sampler.probs_dtype
    self.token_id_dtype = self.spec_decode_sampler.token_id_dtype
    # Lazy initialization.
    self.scorer: SpeculativeScorer
    self.disable_mqa_scorer = False

    # Hidden states from target model to pass to proposer
    # in the subsequent step.
    self.previous_hidden_states: Optional[HiddenStates] = None
    self._disable_logprobs = disable_logprobs
    self._disable_log_stats = disable_log_stats
    self._num_spec_prefill_steps = num_spec_prefill_steps

# msadapter does not support to slice tensor with empty index,
# rewrite this method to optimize the performance(almost 2ms)
@nvtx_range("spec_decode_worker._verify_tokens")
def _verify_tokens(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        proposal_scores: SpeculativeScores,
        proposals: SpeculativeProposals,
        max_proposal_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Determine which speculative tokens are accepted using the
    probabilities of each token according to the proposer and scorer models.

    Returns a tuple of Tensors, one for the accepted token ids and one for
    the logprobs according to the scoring model.
    """
    proposal_lens_list = proposals.proposal_lens.tolist()

    # vLLM currently only supports proposal lens equal to zero or the batch
    # proposal len. This adds some complexity (splitting the batch into spec
    # and non spec sequences) and should be removed in the future. It can be
    # done by supporting per-sequence proposal lens.
    (_, spec_indices), (_, non_spec_indices) = split_batch_by_proposal_len(
        seq_group_metadata_list, proposal_lens_list)
    original_indices = spec_indices + non_spec_indices

    proposal_verifier_probs = proposal_scores.probs
    bonus_token_ids = proposal_scores.token_ids[:, -1:]
    proposal_probs = proposals.proposal_probs
    proposal_token_ids = proposals.proposal_token_ids
    if non_spec_indices:
        # Get probabilities of target model, including bonus tokens.
        proposal_verifier_probs = proposal_verifier_probs[spec_indices]
        # Get bonus tokens from target model.
        bonus_token_ids = bonus_token_ids[spec_indices]
        # Get probabilities according to proposal method.
        proposal_probs = proposal_probs[spec_indices]
        # Get proposed tokens.
        proposal_token_ids = proposal_token_ids[spec_indices]

    # Sampler arguments
    sampler_extra_kwargs: Dict[str, Any] = {}
    if self.generators and isinstance(self.spec_decode_sampler,
                                      SpecDecodeStochasticBaseSampler):
        sampler_extra_kwargs["seeded_seqs"] = {
            idx: self.generators[sgm.request_id]
            for idx, sgm in enumerate(seq_group_metadata_list)
            if sgm.sampling_params.seed is not None
        }

    accepted_token_ids = self.spec_decode_sampler(
        target_with_bonus_probs=proposal_verifier_probs,
        bonus_token_ids=bonus_token_ids,
        draft_probs=proposal_probs,
        draft_token_ids=proposal_token_ids,
        **sampler_extra_kwargs,
    )
    if non_spec_indices:
        # Get non-speculative sampled tokens from target model.
        non_spec_token_ids = proposal_scores.token_ids[non_spec_indices].expand(-1, max_proposal_len + 1).clone()

        # Append output tokens from non-speculative sequences to
        # the accepted token ids tensor.
        non_spec_token_ids[:, 1:] = -1
        accepted_token_ids = torch.cat([accepted_token_ids, non_spec_token_ids])
        # Rearrange so that results are in the order of the original seq group
        # metadata.
        accepted_token_ids[original_indices] = accepted_token_ids.clone()

    logprobs = proposal_scores.logprobs
    # B x K+1 x D
    hidden_states = proposal_scores.hidden_states
    if hidden_states is not None:
        # Only get terminal hidden states for next step
        terminal_metadata = [
            sg for sg in seq_group_metadata_list if sg.do_sample
        ]

        # Contract hidden states based on accepted tokens
        hs_size = hidden_states.shape[-1]
        accepted_index = accepted_token_ids + 1  # Convert -1 to 0
        accepted_index = accepted_index.count_nonzero(dim=1).add_(-1)  # b
        # Drop non-terminal prefill chunks hidden states.
        if VLLM_INVALID_TOKEN_ID in accepted_index.tolist():
            hidden_states = hidden_states[accepted_index != VLLM_INVALID_TOKEN_ID]
            accepted_index = accepted_index[accepted_index != VLLM_INVALID_TOKEN_ID]
        assert len(accepted_index) == hidden_states.shape[0] == len( terminal_metadata)
        index = accepted_index[:, None, None].expand(-1, 1, hs_size)  # b x 1 x d
        second_last_token_hidden_states = hidden_states[:, -2]  # b x d
        hidden_states = hidden_states.gather(1, index).squeeze(1)  # b x d
        # Store hidden states from target model for subsequent decode step
        self.previous_hidden_states = HiddenStates(
            hidden_states, terminal_metadata,
            second_last_token_hidden_states)
    return accepted_token_ids, logprobs


from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.spec_decode.spec_decode_worker import prepare_prefill_hidden_states

# the 'where' ops in msadapter does not support condition-only inputs, use nonzero
@nvtx_range("spec_decode_worker._run_no_spec")
def _run_no_spec(self, execute_model_req: ExecuteModelRequest,
                 skip_proposer: bool) -> List[SamplerOutput]:
    """Run a single generation step without any speculation. The input is
    sent to the proposer and scorer model so that the KV cache is consistent
    between the two. When skip_proposer is True, the proposer model is
    not called, meaning that the kv-cache in proposer for requests is not
    updated, so they cannot enable spec decode in the rest decoding.
    """

    sampler_output = self.scorer_worker.execute_model(execute_model_req)
    assert len(sampler_output) == 1
    sampler_output = sampler_output[0]

    # Store hidden states from target model execution, BxD.
    hidden_states = sampler_output.hidden_states
    if hidden_states is not None:
        # Only decodes and prefill terminal chunks need a hidden state.
        seq_group_meta_with_hidden = [
            sg for sg in execute_model_req.seq_group_metadata_list
            if sg.do_sample
        ]
        if any(seq.is_prompt for seq in seq_group_meta_with_hidden):
            # Drop hidden_states with no prediction (eg non-terminal chunks)
            hidden_states = hidden_states[
                (sampler_output.sampled_token_ids - VLLM_INVALID_TOKEN_ID).nonzero(as_tuple=True)[0]]
        if self.previous_hidden_states is None and len(
                seq_group_meta_with_hidden):
            self.previous_hidden_states = HiddenStates(
                hidden_states, seq_group_meta_with_hidden)
        elif self.previous_hidden_states and len(
                seq_group_meta_with_hidden):
            self.previous_hidden_states.update(hidden_states,
                                               seq_group_meta_with_hidden)

    if not skip_proposer:
        # We prepare the prefill hidden states here so that there no
        # additional complexity in worker for spec_decode vs non_spec_decode
        # flow and execute_model doesn't need additional modifications.
        execute_model_req.previous_hidden_states = \
            prepare_prefill_hidden_states(
                sampler_output.prefill_hidden_states)
        for i in range(self._num_spec_prefill_steps):
            execute_model_req.spec_step_idx = i
            self.proposer_worker.execute_model(execute_model_req)

    sampler_output_to_return = (self._serialize_sampler_output_no_logprobs(
        execute_model_req=execute_model_req, sampler_output=sampler_output)
                                if self._disable_logprobs else
                                [sampler_output])

    # Clear device tensors from sampler output. This reduces communication
    # overhead when the engine runs in a different process than the workers.
    sampler_output.sampled_token_probs = None
    sampler_output.sampled_token_ids = None
    sampler_output.logprobs = None
    return sampler_output_to_return


# the output of 'tensor.max()' does not consistent with torch
def _create_output(
        self,
        accepted: torch.Tensor,  # [batch_size, k]
        substitute_token_ids: torch.Tensor,  # [batch_size, k]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
        bonus_token_ids: torch.Tensor,  # [batch_size]
) -> torch.Tensor:
    """Format output. Returns a matrix of token ids. When
    a token is rejected via sampling, all subsequent token ids are
    set to -1 for the sequence.

    Args:
        accepted: A boolean tensor indicating if the corresponding
        draft token in draft_token_ids should be accepted or not.
        substitute_token_ids: A tensor of token_ids that can be used
        as substitutes for the draft token ids if the proposed token
        is rejected.
        draft_token_ids: A tensor of token ids speculated by the
        draft model.
        bonus_token_ids: Token ids to use as the bonus token if
        all the draft tokens are accepted.
    Returns:
        A tensor containing the accepted token ids. The shape of the
        tensor is [batch_size, k + num_bonus_tokens]
    """
    # the return type of max is a tuple in msadapter
    batch_size, k = substitute_token_ids.shape
    assert self._num_bonus_tokens == 1    # ToDo: only support 1 mtp layer to optimize performance(almost 2ms)

    # Create an extended output tensor
    output_with_bonus_tokens = -torch.ones(
        (batch_size, k + self._num_bonus_tokens),
        dtype=self.token_id_dtype,
        device=accepted.device)

    # Fill in the first k columns of the output tensor using masks and data tensors.
    output_with_bonus_tokens[:, :k] = draft_token_ids * accepted + substitute_token_ids * (~accepted)

    # Fill the last column.
    # We check output directly as accepted may have True values inconsistentwith causal acceptance.
    # Fill the recovered token ids.
    output_with_bonus_tokens[:, -1:] = bonus_token_ids * accepted + (-1) * (~accepted)

    self.num_accepted_tokens += accepted.sum()
    self.num_emitted_tokens += (output_with_bonus_tokens != -1).sum()
    self.num_draft_tokens += batch_size * k

    return output_with_bonus_tokens


# msadapter does not support 'new_full', and the operator 'new_zero' only supports a list or a tuple as an input
from vllm.spec_decode.util import sampler_output_to_torch
def _merge_outputs(
        self,
        batch_size: int,
        proposal_len: int,
        maybe_sampler_output: Optional[List[SamplerOutput]],
        proposal_lens: List[int],
        nonzero_proposal_len_indices: List[int],
        sampler_transposed: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """After speculations are produced, merge the speculation results with
    the skipped sequences.
    """
    if maybe_sampler_output is None:
        # If no speculative tokens, the sampler output will be None.
        # In this case we return empty proposals.
        proposal_tokens = torch.tensor(-1,
                                       dtype=torch.long,
                                       device=self._device).expand(
            batch_size, proposal_len)
        proposal_probs = torch.tensor(0,
                                      dtype=torch.float32,
                                      device=self._device).expand(
            batch_size, proposal_len,
            self._vocab_size)
        proposal_lens_tensor = torch.tensor(0,
                                            dtype=torch.long,
                                            device=self._device).expand(
            len(proposal_lens))
        return proposal_tokens, proposal_probs, proposal_lens_tensor

    sampler_output = maybe_sampler_output
    proposal_tokens, proposal_probs, *_ = sampler_output_to_torch(
        sampler_output, sampler_transposed)

    # Now, reformat the output GPU tensors such that each sequence has
    # a proposal. the proposal can be empty, e.g. [-1, -1, -1]

    # entire_proposal_tokens = proposal_tokens.new_full(
    #     size=(batch_size, *proposal_tokens.shape[1:]),
    #     fill_value=-1,
    # )
    entire_proposal_tokens = torch.full(size=(batch_size, *proposal_tokens.shape[1:]), fill_value=-1)
    entire_proposal_tokens[nonzero_proposal_len_indices] = proposal_tokens
    entire_proposal_probs = proposal_probs.new_zeros((
        batch_size,
        *proposal_probs.shape[1:],)
    )
    entire_proposal_probs[nonzero_proposal_len_indices] = proposal_probs

    proposal_tokens, proposal_probs = (
        entire_proposal_tokens,
        entire_proposal_probs,
    )

    proposal_lens_tensor = torch.zeros(batch_size,
                                       dtype=torch.long,
                                       device=self._device)
    proposal_lens_tensor[nonzero_proposal_len_indices] = proposal_len

    return proposal_tokens, proposal_probs, proposal_lens_tensor
