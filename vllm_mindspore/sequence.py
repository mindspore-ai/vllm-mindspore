# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/sequence.py
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
"""Sequence and its related classes."""
import copy
import enum
from abc import ABC, abstractmethod
from array import array
from collections import defaultdict
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Callable, DefaultDict, Dict, List, Mapping, Optional
from typing import Sequence as GenericSequence
from typing import Set, Tuple, Union

import msgspec
import torch

from vllm.inputs import SingletonInputs, SingletonInputsAdapter
from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalDataDict, MultiModalPlaceholderDict
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import RequestOutputKind, SamplingParams

VLLM_TOKEN_ID_ARRAY_TYPE = "l"

VLLM_INVALID_TOKEN_ID = -1


def array_full(token_id: int, count: int):
    """:class:`array` equivalent of :func:`numpy.full`."""
    return array(VLLM_TOKEN_ID_ARRAY_TYPE, [token_id]) * count


# We use dataclass for now because it is used for
# openai server output, and msgspec is not serializable.
# TODO(sang): Fix it.
@dataclass
class Logprob:
    """Infos for supporting OpenAI compatible logprobs and token ranks.

    Attributes:
        logprob: The logprob of chosen token
        rank: The vocab rank of chosen token (>=1)
        decoded_token: The decoded chosen token index
    """
    logprob: float
    rank: Optional[int] = None
    decoded_token: Optional[str] = None


# {token_id -> logprob} per each sequence group. None if the corresponding
# sequence group doesn't require prompt logprob.
PromptLogprobs = List[Optional[Dict[int, Logprob]]]
# {token_id -> logprob} for each sequence group.
SampleLogprobs = List[Dict[int, Logprob]]


class SequenceStatus(enum.IntEnum):
    """Status of a sequence."""
    WAITING = 0
    RUNNING = 1
    SWAPPED = 2
    # Note: anything after SWAPPED (2) will be considered
    # as a finished status.
    FINISHED_STOPPED = 3
    FINISHED_LENGTH_CAPPED = 4
    FINISHED_ABORTED = 5
    FINISHED_IGNORED = 6

    @staticmethod
    def is_finished(status: "SequenceStatus") -> bool:
        return status > SequenceStatus.SWAPPED

    @staticmethod
    def get_finished_reason(status: "SequenceStatus") -> Union[str, None]:
        if status == SequenceStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == SequenceStatus.FINISHED_LENGTH_CAPPED:
            finish_reason = "length"
        elif status == SequenceStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == SequenceStatus.FINISHED_IGNORED:
            # The ignored sequences are the sequences whose prompt lengths
            # are longer than the model's length cap. Therefore, the stop
            # reason should also be "length" as in OpenAI API.
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason


class SequenceStage(enum.Enum):
    PREFILL = enum.auto()
    DECODE = enum.auto()


@dataclass
class RequestMetrics:
    """Metrics associated with a request.

    Attributes:
        arrival_time: The time when the request arrived.
        first_scheduled_time: The time when the request was first scheduled.
        first_token_time: The time when the first token was generated.
        time_in_queue: The time the request spent in the queue.
        finished_time: The time when the request was finished.
        scheduler_time: The time spent in the scheduler when this request was
                        being considered by the scheduler.
        model_forward_time: The time spent in the model forward pass when this
                            request was in the batch.
        model_execute_time: The time spent in the model execute function. This
                            will include model forward, block/sync across
                            workers, cpu-gpu sync time and sampling time.
    """
    arrival_time: float
    last_token_time: float
    first_scheduled_time: Optional[float]
    first_token_time: Optional[float]
    time_in_queue: Optional[float]
    finished_time: Optional[float] = None
    scheduler_time: Optional[float] = None
    model_forward_time: Optional[float] = None
    model_execute_time: Optional[float] = None


class SequenceDataDelta(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True):  # type: ignore[call-arg]
    """Delta SequenceData to send to workers per step."""
    # A new token to be appended to existing SequenceData.
    new_output_token_ids: List[int]
    # Overwriting existing `cumulative_logprob`
    new_cumulative_logprob: float
    # Overwriting existing `num_computed_tokens`.
    new_num_computed_tokens: int
    # Overwriting existing `stage`.
    new_stage: SequenceStage


class SequenceData(msgspec.Struct,
                   omit_defaults=True):  # type: ignore[call-arg]
    """Data associated with a sequence.

    Args:
        prompt_token_ids: The token IDs of the prompt.
        output_token_ids: The token IDs of the output. Set to an empty list if
            None.

    Attributes:
        prompt_token_ids: The token IDs of the prompt.
        output_token_ids: The token IDs of the output.
        cumulative_logprob: The cumulative log probability of the output.
    """
    # NOTE: we cannot use Union[List, array] because msgspec cannot support
    # union of 2 list types.
    _prompt_token_ids: array
    _output_token_ids: array = msgspec.field(
        default_factory=lambda: array(VLLM_TOKEN_ID_ARRAY_TYPE, []))

    ### The below fields should not be passed as an argument ###
    _cumulative_logprob: float = 0.0
    _prompt_token_ids_tuple: Tuple[int,
                                   ...] = msgspec.field(default_factory=tuple)
    # The number of tokens that are computed (that run against the model).
    _num_computed_tokens: int = 0
    # The number of tokens with prefix cache hit.
    _num_cached_tokens: int = 0
    _stage: SequenceStage = SequenceStage.PREFILL
    _cached_all_token_ids: List[int] = msgspec.field(default_factory=list)

    # It is used to get delta input. It is reset when `get_delta_and_reset`
    # is called.
    _new_appended_tokens: List[int] = msgspec.field(default_factory=list)

    # It is used to compute mrope_position_ids.
    _mrope_position_delta: Optional[int] = None

    @staticmethod
    def from_prompt_token_counts(
            *token_counts: Tuple[int, int]) -> "SequenceData":
        """
        Construct a :class:`SequenceData` instance by concatenating
        prompt token sequences.

        Each tuple represents one token sequence, expressed in the form
        :code:`(token_id, count)`.
        """
        if len(token_counts) == 0:
            return SequenceData.from_seqs([])

        prompt_token_ids_arr = reduce(
            array.__iadd__,
            (array_full(token_id, count) for token_id, count in token_counts),
        )

        return SequenceData(prompt_token_ids_arr)

    @staticmethod
    def from_seqs(
        prompt_token_ids: GenericSequence[int],
        output_token_ids: Optional[GenericSequence[int]] = None,
    ) -> "SequenceData":
        """
        Construct a :class:`SequenceData` instance from prompt and output
        token sequences.
        """
        prompt_token_ids_arr = array(VLLM_TOKEN_ID_ARRAY_TYPE,
                                     prompt_token_ids)

        if output_token_ids is None:
            return SequenceData(prompt_token_ids_arr)

        output_token_ids_arr = array(VLLM_TOKEN_ID_ARRAY_TYPE,
                                     output_token_ids)

        return SequenceData(prompt_token_ids_arr,
                            _output_token_ids=output_token_ids_arr)

    def __post_init__(self) -> None:
        assert self._prompt_token_ids.typecode == "l"
        assert self._output_token_ids.typecode == "l"
        self._prompt_token_ids_tuple: Tuple[int, ...] = tuple(
            self._prompt_token_ids)
        self._update_cached_all_tokens()

    def _update_cached_all_tokens(self):
        assert isinstance(self._prompt_token_ids, array)
        assert isinstance(self._output_token_ids, array)
        self._cached_all_token_ids: List[int] = list(self._prompt_token_ids +
                                                     self._output_token_ids)

    @property
    def cumulative_logprob(self) -> float:
        return self._cumulative_logprob

    @property
    def prompt_token_ids(self) -> Tuple[int, ...]:
        return self._prompt_token_ids_tuple

    @prompt_token_ids.setter
    def prompt_token_ids(self, new_prompt_token_ids) -> None:
        raise NotImplementedError

    @property
    def prompt_token_ids_array(self) -> array:
        """Return the prompt token ids in array type.

        Note that the array is in "I" type, and it is not compatible
        with torch.long (2 bytes vs 4 bytes). So beware of the usage.
        """
        return self._prompt_token_ids

    @property
    def output_token_ids(self) -> Tuple[int, ...]:
        return tuple(self._output_token_ids)

    @output_token_ids.setter
    def output_token_ids(self,
                         new_output_token_ids: GenericSequence[int]) -> None:
        self._output_token_ids = array(VLLM_TOKEN_ID_ARRAY_TYPE,
                                       new_output_token_ids)
        self._update_cached_all_tokens()

    @property
    def output_token_ids_array(self) -> array:
        """Return the prompt token ids in array type.

        Note that the array is in "I" type, and it is not compatible
        with torch.long (2 bytes vs 4 bytes). So beware of the usage.
        """
        assert isinstance(self._output_token_ids, array)
        return self._output_token_ids

    @property
    def mrope_position_delta(self) -> Optional[int]:
        return self._mrope_position_delta

    @mrope_position_delta.setter
    def mrope_position_delta(self, new_mrope_position_delta):
        self._mrope_position_delta = new_mrope_position_delta

    def append_token_id(self, token_id: int, logprob: float) -> None:
        self._output_token_ids.append(token_id)
        self._new_appended_tokens.append(token_id)
        self._cached_all_token_ids.append(token_id)
        self._cumulative_logprob += logprob

    def get_len(self) -> int:
        return len(self._output_token_ids) + len(self._prompt_token_ids)

    def get_prompt_len(self) -> int:
        return len(self._prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self._output_token_ids)

    def get_token_ids(self) -> List[int]:
        return self._cached_all_token_ids

    def get_prefix_token_ids(
            self, num_tokens: int
    ) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        """Get prefix tokens, and make the return value hashable"""
        prompt_length = self.get_prompt_len()
        if num_tokens > prompt_length:
            return (self._prompt_token_ids_tuple,
                    tuple(self._output_token_ids[:num_tokens - prompt_length]))
        else:
            return (self._prompt_token_ids_tuple[:num_tokens], None)

    def get_num_computed_tokens(self) -> int:
        """Return the number of prefill tokens that are already computed."""
        return self._num_computed_tokens

    def update_num_computed_tokens(self, num_new_computed_tokens: int):
        """Update number of tokens computed so far."""
        self._num_computed_tokens += num_new_computed_tokens
        assert self._num_computed_tokens <= self.get_len(), (
            self._num_computed_tokens, self.get_len())
        # If all tokens are computed, it means it is in decoding phase.
        if self.get_num_uncomputed_tokens() == 0:
            self._stage = SequenceStage.DECODE

    def get_num_cached_tokens(self) -> int:
        """Return the number of tokens with prefix cache hit."""
        return self._num_cached_tokens

    def update_num_cached_tokens(self, num_cached_tokens: int):
        """Update the number of tokens with prefix cache hit."""
        self._num_cached_tokens = num_cached_tokens

    def reset_state_for_recompute(self) -> None:
        """Reset the number of computed tokens from this sequence. It is
        supposed to be called when a sequence needs to be started from
        the beginning again (e.g., sequence is preempted).
        """
        self._num_computed_tokens = 0
        self._stage = SequenceStage.PREFILL
        self._new_appended_tokens = []

    def get_num_uncomputed_tokens(self) -> int:
        """Return the number of prefill tokens that are not computed."""
        # we use `get_len()` which includes prompt_len + output_len instead
        # of prompt_len here. This is because during recompute we need to
        # prefill for both prompt and output.
        return self.get_len() - self.get_num_computed_tokens()

    def get_last_token_id(self) -> int:
        if not self._output_token_ids:
            return self._prompt_token_ids[-1]
        return self._output_token_ids[-1]

    def get_prompt_token_ids(self) -> Tuple[int, ...]:
        return self.prompt_token_ids

    def get_output_token_ids(self) -> Tuple[int, ...]:
        return self.output_token_ids

    def get_delta_and_reset(self) -> SequenceDataDelta:
        delta = SequenceDataDelta(self._new_appended_tokens,
                                  self._cumulative_logprob,
                                  self.get_num_computed_tokens(), self.stage)
        # Reset delta state.
        self._new_appended_tokens = []
        return delta

    def apply_delta(self, delta: SequenceDataDelta):
        self._num_computed_tokens = delta.new_num_computed_tokens
        self._cumulative_logprob = delta.new_cumulative_logprob
        self._stage = delta.new_stage
        self._output_token_ids.extend(delta.new_output_token_ids)
        self._cached_all_token_ids.extend(delta.new_output_token_ids)

    @property
    def stage(self) -> SequenceStage:
        return self._stage

    def __repr__(self) -> str:
        return (f"SequenceData("
                f"prompt_token_ids={self._prompt_token_ids}, "
                f"output_token_ids={self.output_token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob}, "
                f"get_num_computed_tokens={self.get_num_computed_tokens()}")


class Sequence:
    """Stores the data, status, and block information of a sequence.

    The sequence is constructed from the :data:`DecoderOnlyInputs`
    (for decoder-only) or :data:`EncoderDecoderInputs` (for encoder-decoder)
    instance passed in through the :code:`inputs` constructor argument.

    Args:
        seq_id: The ID of the sequence.
        inputs: The inputs of the sequence.
        block_size: The block size of the sequence. Should be the same as the
            block size used by the block manager and cache engine.
        eos_token_id: The end-of-sequence (EOS) token id recognized by this LLM.
        lora_request: LoRA request.
        prompt_adapter_request: Prompt Adapter request.
    """

    def __init__(
        self,
        seq_id: int,
        inputs: SingletonInputs,
        block_size: int,
        eos_token_id: Optional[int] = None,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> None:
        self.seq_id = seq_id
        self.inputs = SingletonInputsAdapter(inputs)
        self.block_size = block_size
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request
        self.prompt_adapter_request = prompt_adapter_request

        self.data = SequenceData.from_seqs(self.prompt_token_ids)
        self.output_logprobs: SampleLogprobs = []
        self.output_text = ""

        self.status = SequenceStatus.WAITING
        self.stop_reason: Union[int, str, None] = None

        # These are used to keep track of delta outputs
        self._last_output_token_ids_offset: int = 0
        self._last_output_text_offset: int = 0

        # Used for incremental detokenization
        self.prefix_offset = 0
        self.read_offset = 0
        # Input + output tokens
        self.tokens: Optional[List[str]] = None

    @property
    def n_blocks(self) -> int:
        return (self.get_len() + self.block_size - 1) // self.block_size

    @property
    def prompt(self) -> Optional[str]:
        return self.inputs.prompt

    @property
    def prompt_token_ids(self) -> List[int]:
        return self.inputs.prompt_token_ids

    @property
    def prompt_embeds(self) -> Optional[torch.Tensor]:
        return self.inputs.prompt_embeds

    @property
    def token_type_ids(self) -> List[int]:
        return self.inputs.token_type_ids

    @property
    def multi_modal_data(self) -> "MultiModalDataDict":
        return self.inputs.multi_modal_data

    @property
    def multi_modal_placeholders(self) -> MultiModalPlaceholderDict:
        return self.inputs.multi_modal_placeholders

    @property
    def mm_processor_kwargs(self) -> Dict[str, Any]:
        return self.inputs.mm_processor_kwargs

    @property
    def lora_int_id(self) -> int:
        return self.lora_request.lora_int_id if self.lora_request else 0

    @property
    def prompt_adapter_id(self) -> int:
        return self.prompt_adapter_request.prompt_adapter_id \
                        if self.prompt_adapter_request else 0

    def get_output_text_to_return(self, buffer_length: int,
                                  delta: bool) -> str:
        """If delta is True, only new text since the last call to
        this method is returned"""

        # We return the full output text if the sequence is finished.
        truncate = buffer_length and not self.is_finished()
        if not delta:
            return self.output_text[:-buffer_length] if truncate else (
                self.output_text)
        length = len(self.output_text)
        if truncate:
            length -= buffer_length
        last_offset = self._last_output_text_offset
        if last_offset < length:
            self._last_output_text_offset = length
            return self.output_text[last_offset:length]
        return ""

    def get_output_token_ids_to_return(
            self, delta: bool) -> Union[GenericSequence[int], int]:
        """If delta is True, only new tokens since the last call to
        this method are returned"""
        if not delta:
            return self.get_output_token_ids()

        output_len = self.get_output_len()

        # Get the number of new tokens
        num_new_tokens = output_len - self._last_output_token_ids_offset
        self._last_output_token_ids_offset = output_len

        # Return new tokens
        if num_new_tokens == 1:
            # Optimization for single decode token case
            # (which is what we have most of the time)
            return self.data._cached_all_token_ids[-1]

        if num_new_tokens == 0:
            return []

        return self.data._cached_all_token_ids[-num_new_tokens:]

    def hash_of_block(self, logical_idx: int) -> int:
        # TODO This can produce incorrect hash when block size > prompt size

        # Compute the number of tokens in the sequence
        # TODO: The current hashing function is O(L^2). We should optimize
        # this in the future.
        num_tokens = self.num_hashed_tokens_of_block(logical_idx)
        hashed_tokens = self.data.get_prefix_token_ids(num_tokens)
        return hash((hashed_tokens, self.lora_int_id))

    def extra_hash(self) -> Optional[int]:
        """
        This function computes an extra hash for a sequence, specifically
        designed for prefix caching mode. The final sequence hash is determined
        by applying token_ids from the sequence's blocks.
        """
        if self.prompt_adapter_id == 0 and self.lora_int_id == 0:
            return None

        # NOTE: If there are additional factors influencing the block aside from
        # token_ids, include them as input parameters to the hash.
        return hash((self.prompt_adapter_id, self.lora_int_id))

    def num_hashed_tokens_of_block(self, logical_idx: int):
        return logical_idx * self.block_size + self.block_size

    def reset_state_for_recompute(self):
        """Reset the sequence states for recomputation."""
        self.data.reset_state_for_recompute()

    def append_token_id(self, token_id: int, logprobs: Dict[int,
                                                            Logprob]) -> None:
        assert token_id in logprobs
        self.output_logprobs.append(logprobs)
        self.data.append_token_id(token_id, logprobs[token_id].logprob)

    def get_len(self) -> int:
        return self.data.get_len()

    def get_prompt_len(self) -> int:
        return self.data.get_prompt_len()

    def get_output_len(self) -> int:
        return self.data.get_output_len()

    def get_token_ids(self) -> List[int]:
        return self.data.get_token_ids()

    def get_prompt_token_ids(self) -> Tuple[int, ...]:
        return self.data.get_prompt_token_ids()

    def get_last_token_id(self) -> int:
        return self.data.get_last_token_id()

    def get_output_token_ids(self) -> Tuple[int, ...]:
        return self.data.get_output_token_ids()

    def get_cumulative_logprob(self) -> float:
        return self.data.cumulative_logprob

    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self.status)

    def fork(self, new_seq_id: int) -> "Sequence":
        new_seq = copy.deepcopy(self)
        new_seq.seq_id = new_seq_id
        return new_seq

    def get_num_new_tokens(self) -> int:
        """Get the number of new tokens to be computed.

        Returns:
            The new number of tokens to be computed. I.e., 1 for decode, or
            the remaining prompt size for prefill.
        """
        if self.data.stage == SequenceStage.DECODE:
            return 1
        return self.data.get_num_uncomputed_tokens()

    def get_num_computed_tokens(self) -> int:
        return self.data.get_num_computed_tokens()

    def is_prefill(self) -> bool:
        return self.data.stage == SequenceStage.PREFILL

    def __repr__(self) -> str:
        return (f"Sequence(seq_id={self.seq_id}, "
                f"status={self.status.name}, "
                f"num_blocks={self.n_blocks}, ")


class SequenceGroupState(msgspec.Struct,
                         omit_defaults=True):  # type: ignore[call-arg]
    """Mutable state tied to a specific sequence group"""

    # for multi-step decoding
    num_steps: int = 1
    current_step: int = 0

    @property
    def remaining_steps(self) -> int:
        return self.num_steps - self.current_step


class SequenceGroup:
    """A group of sequences that are generated from the same prompt.

    Args:
        request_id: The ID of the request.
        seqs: The list of sequences.
        sampling_params: The sampling parameters used to generate the outputs.
        arrival_time: The arrival time of the request.
        lora_request: LoRA request.
        pooling_params: The parameters used to generate the pooler
            for a pooling model.
        pooled_data: The extracted hidden states from a pooling model.
        encoder_seq: Optional, the single encoder sequence. Should be None
                     unless you are working with an encoder/decoder model.
        trace_headers: OpenTelemetry trace headers.
        prompt_adapter_request: Prompt Adapter request.
        priority: User-defined priority of the request.
    """

    def __init__(
        self,
        request_id: str,
        seqs: List[Sequence],
        arrival_time: float,
        sampling_params: Optional[SamplingParams] = None,
        lora_request: Optional[LoRARequest] = None,
        pooling_params: Optional[PoolingParams] = None,
        pooled_data: Optional[torch.Tensor] = None,
        encoder_seq: Optional[Sequence] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> None:
        self.request_id = request_id
        self.seqs = seqs
        self.first_seq = seqs[0]
        self.arrival_time = arrival_time
        self.is_single_seq = len(seqs) == 1
        self.seqs_dict = {seq.seq_id: seq for seq in seqs}

        self.sampling_params = sampling_params
        self.metrics = RequestMetrics(arrival_time=arrival_time,
                                      last_token_time=arrival_time,
                                      first_scheduled_time=None,
                                      first_token_time=None,
                                      time_in_queue=None)
        self.lora_request = lora_request
        self.prompt_logprobs: Optional[PromptLogprobs] = None
        self.state = SequenceGroupState()
        self.pooling_params = pooling_params
        self.pooled_data = pooled_data
        self.prompt_adapter_request = prompt_adapter_request
        self.encoder_seq = encoder_seq
        self.trace_headers = trace_headers
        self.priority = priority

        self.cached_request_output = None

    @property
    def prompt(self) -> Optional[str]:
        return self.first_seq.prompt

    @property
    def prompt_token_ids(self) -> List[int]:
        return self.first_seq.prompt_token_ids

    @property
    def encoder_prompt(self) -> Optional[str]:
        # There are either 0 or 1 encoder sequences
        # If one is present, its prompt is distinct
        # from the decoder's.
        return (self.encoder_seq.prompt
                if self.encoder_seq is not None else None)

    @property
    def encoder_prompt_token_ids(self) -> Optional[List[int]]:
        # There are either 0 or 1 encoder sequences
        # If one is present, its prompt token ids are
        # distinct from the decoder's.
        return (self.encoder_seq.prompt_token_ids
                if self.encoder_seq is not None else None)

    @property
    def token_type_ids(self) -> Optional[List[int]]:
        return self.first_seq.token_type_ids

    @property
    def multi_modal_data(self) -> MultiModalDataDict:
        return self.first_seq.multi_modal_data

    @property
    def multi_modal_placeholders(self) -> MultiModalPlaceholderDict:
        return self.first_seq.multi_modal_placeholders

    @property
    def mm_processor_kwargs(self) -> Dict[str, Any]:
        return self.first_seq.mm_processor_kwargs

    @property
    def lora_int_id(self) -> int:
        return self.lora_request.lora_int_id if self.lora_request else 0

    @property
    def prompt_adapter_id(self) -> int:
        return self.prompt_adapter_request.prompt_adapter_id \
                        if self.prompt_adapter_request else 0

    @property
    def prompt_adapter_num_virtual_tokens(self) -> int:
        return self.prompt_adapter_request.prompt_adapter_num_virtual_tokens\
                         if self.prompt_adapter_request else 0

    def init_multi_step(self, num_steps: int) -> None:
        self.state.num_steps = num_steps
        self.state.current_step = 0

    def init_multi_step_from_lookahead_slots(self, num_lookahead_slots: int,
                                             num_scheduler_steps: int,
                                             is_multi_step: bool,
                                             enable_chunking: bool) -> None:

        if not is_multi_step:
            self.init_multi_step(num_steps=num_scheduler_steps)
            return

        # Multi-Step case
        is_prefill = self.is_prefill()

        # The asserts below reflect the expectations of the current system.
        if is_prefill and enable_chunking:
            assert num_lookahead_slots == num_scheduler_steps
            self.init_multi_step(num_steps=num_lookahead_slots)
        else:
            is_decode: bool = not is_prefill
            # If it is a prefill, num_lookahead_slots must be 0
            assert num_lookahead_slots == 0 or is_decode
            # If it is a decode, num_lookahead_slots + 1 must match
            # the scheduler steps.
            assert num_lookahead_slots + 1 == num_scheduler_steps or is_prefill
            self.init_multi_step(num_steps=num_lookahead_slots + 1)

    def get_last_latency(self, now: float) -> float:
        """Sets the last token time for Request level timings."""
        # If still in prefill phase, raise Error.
        if self.is_prefill():
            raise ValueError(
                "seq_group.get_last_latency() should not be called "
                "if the seq_group is in prefill phase.")

        # Otherwise return token latency.
        latency = now - self.metrics.last_token_time
        self.metrics.last_token_time = now
        return latency

    def maybe_set_first_token_time(self, time: float) -> None:
        """Sets the first token time for Request level timings."""
        # Note: in a case where a sequence_group is swapped and
        #   recomputed, the time between iterations is counted
        #   in TPOT, rather than recalculating TTFT (since from the )
        #   POV of the user, there is simply a long generation delay.
        if (self.metrics.first_token_time is None
                and self.first_seq.get_output_len() == 1):
            self.metrics.first_token_time = time

    def maybe_set_first_scheduled_time(self, time: float) -> None:
        """Sets the first scheduled time and time in queue for Request
        level timings."""
        if self.metrics.first_scheduled_time is None:
            self.metrics.first_scheduled_time = time
            self.metrics.time_in_queue = time - self.metrics.arrival_time

    def set_finished_time(self, time: Optional[float]) -> None:
        """Sets the finished time for Request level timings."""
        self.metrics.finished_time = time

    def get_max_num_running_seqs(self) -> int:
        """The maximum number of sequences running in parallel in the remaining
        lifetime of the request."""
        return 0 if self.first_seq.is_finished() else 1

    def get_seqs(
        self,
        status: Optional[SequenceStatus] = None,
    ) -> List[Sequence]:
        if status is None:
            return self.seqs

        return self.seqs if self.first_seq.status == status else []

    def is_encoder_decoder(self) -> bool:
        return self.encoder_seq is not None

    def get_encoder_seq(self) -> Optional[Sequence]:
        return self.encoder_seq

    def get_finished_seqs(self) -> List[Sequence]:
        return self.seqs if self.first_seq.is_finished() else []

    def update_num_computed_tokens(self, num_new_computed_tokens: int):
        """Update number of tokens computed so far."""
        seq = self.first_seq
        if not seq.is_finished():
            seq.data.update_num_computed_tokens(num_new_computed_tokens)

    def get_num_uncomputed_tokens(self) -> int:
        num_uncomputed_tokens = 0
        seq = self.first_seq
        if not seq.is_finished():
            num_uncomputed_tokens += seq.data.get_num_uncomputed_tokens()
        return num_uncomputed_tokens

    def num_seqs(self, status: Optional[SequenceStatus] = None) -> int:
        # Optimization. We don't need to call get_seqs if we don't need to
        # filter by states.
        if status is None:
            return len(self.seqs)

        if self.is_single_seq:
            return 1 if self.seqs[0].status == status else 0

        return len(self.get_seqs(status))

    def num_finished_seqs(self) -> int:
        return 1 if self.first_seq.is_finished() else 0

    def is_finished(self) -> bool:
        return self.first_seq.is_finished()

    def is_prefill(self) -> bool:
        return self.first_seq.is_prefill()

    def __repr__(self) -> str:
        return (f"SequenceGroup(request_id={self.request_id}, "
                f"sampling_params={self.sampling_params}, "
                f"num_seqs={len(self.seqs)})")


class SequenceGroupMetadataDelta(
        msgspec.Struct,
        tag=True,  # type: ignore[call-arg]
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True):  # type: ignore[call-arg]
    """Delta of SequenceGroupMetadata.

    After sending the first SequenceGroupMetadata, vLLM scheduler
    only sends delta to reduce the data payload size.
    """
    seq_data_delta: Dict[int, SequenceDataDelta]
    request_id: str
    block_tables: Dict[int, List[int]]
    is_prompt: bool
    do_sample: bool = True
    token_chunk_size: Optional[int] = None
    computed_block_nums: Optional[List[int]] = None
    state: Optional[SequenceGroupState] = msgspec.field(
        default_factory=lambda: SequenceGroupState())


class SequenceGroupMetadata(
        msgspec.Struct,
        tag=True,  # type: ignore[call-arg]
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True):  # type: ignore[call-arg]
    """Metadata for a sequence group. Used to create `AttentionMetadata`.

    Args:
        request_id: The ID of the request.
        is_prompt: Whether the request is at prompt stage.
        seq_data: The sequence data. (Seq id -> sequence data)
        sampling_params: The sampling parameters used to generate the outputs.
        block_tables: The block tables. (Seq id -> list of physical block
            numbers)
        do_sample: True if sampling is required. Sampling is not required when
            e.g., prefill is chunked, and the current iteration only computes
            query tokens for prefill, we don't need sampling.
        token_chunk_size: The number of tokens to be processed (per sequence).
            None if chunking is not required.
        lora_request: LoRA request.
        computed_block_nums: The block numbers that are already computed,
            used in prefix caching.
        state: Internal state tied to this sequence group.
        multi_modal_data: Multi modal data.
        mm_processor_kwargs: Multimodal input processor / mapper overrides.
        encoder_seq_data: Optional sequence data for encoder prompt
                          (SequenceGroup.encoder_seq). Should be None
                          unless you are working with an encoder/decoder
                          model.
        cross_block_table: Optional cross-attention block table associated
                           with the encoder prompt
                           (SequenceGroup.encoder_seq). Should be None
                           unless you are working with an encoder/decoder
                           model.
        prompt_adapter_request: Prompt Adapter request.
    """

    request_id: str
    is_prompt: bool
    seq_data: Dict[int, SequenceData]
    sampling_params: Optional[SamplingParams]
    block_tables: Dict[int, List[int]]
    do_sample: bool = True
    pooling_params: Optional[PoolingParams] = None
    lora_request: Optional[LoRARequest] = None
    computed_block_nums: Optional[List[int]] = None
    state: Optional[SequenceGroupState] = msgspec.field(
        default_factory=lambda: SequenceGroupState())
    # "MultiModalDataDict" types. We have to use Any due to msgspec
    # doesn't allow to have union of 2 different dicts.
    token_type_ids: Optional[List[int]] = None
    multi_modal_data: Optional[Any] = None
    multi_modal_placeholders: Optional[MultiModalPlaceholderDict] = None
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    encoder_seq_data: Optional[SequenceData] = None
    cross_block_table: Optional[List[int]] = None
    prompt_adapter_request: Optional[PromptAdapterRequest] = None
    token_chunk_size: Optional[int] = None

    ### Stateful fields that are lazily defined. ###
    # The number of speculative tokens adopted in this request.
    # None means specuative decoding is not used.
    # Zero means speculative decoding is disabled for some reasons.
    # TODO: We should maintain this states out of the sequence group.
    num_speculative_tokens: Optional[int] = None

    def __post_init__(self):
        if self.seq_data is not None and self.token_chunk_size is None:
            if self.is_prompt:
                self.token_chunk_size = next(iter(
                    self.seq_data.values())).get_len()
            else:
                self.token_chunk_size = 1

    @property
    def lora_int_id(self) -> int:
        return self.lora_request.lora_int_id if self.lora_request else 0

    @property
    def prompt_adapter_id(self) -> int:
        return self.prompt_adapter_request.prompt_adapter_id \
                        if self.prompt_adapter_request else 0

    @property
    def prompt_adapter_num_virtual_tokens(self) -> int:
        return self.prompt_adapter_request.prompt_adapter_num_virtual_tokens \
                        if self.prompt_adapter_request else 0

    # Multi-Step Chunked-Prefill property
    @property
    def is_single_step_prompt(self) -> bool:
        # do_sample is true, only when the token_chunk_size matches the
        # num_uncomputed_tokens of the sequence. This indicates that
        # the prompt will finish processing in a single `execute_model`
        # step.
        return self.is_prompt and self.do_sample

    def get_first_seq_id(self) -> int:
        # This is an efficient way of fetching the seq_id when
        # we know this SequenceGroup has only one sequence.
        return next(iter(self.seq_data))

    def apply_delta(self,
                    sequence_group_metadata_delta: SequenceGroupMetadataDelta):
        for id, delta in sequence_group_metadata_delta.seq_data_delta.items():
            self.seq_data[id].apply_delta(delta)
        assert self.request_id == sequence_group_metadata_delta.request_id
        self.block_tables = sequence_group_metadata_delta.block_tables
        self.token_chunk_size = sequence_group_metadata_delta.token_chunk_size
        self.do_sample = sequence_group_metadata_delta.do_sample
        self.is_prompt = sequence_group_metadata_delta.is_prompt

    def finish_step(self) -> None:
        assert self.state is not None
        assert self.state.current_step < self.state.num_steps, \
            f"current step {self.state.current_step}, num_steps {self.state.num_steps}" # noqa
        self.state.current_step += 1


class SequenceOutput(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        array_like=True):  # type: ignore[call-arg]
    """The model output associated with a sequence.

    Args:
        parent_seq_id: The ID of the parent sequence (for forking in beam
            search).
        output_token: The output token ID.
        logprobs: The logprobs of the output token.
            (Token id -> logP(x_i+1 | x_0, ..., x_i))
    """
    parent_seq_id: int
    output_token: int
    logprobs: Dict[int, Logprob]

    def __repr__(self) -> str:
        return (f"SequenceOutput(parent_seq_id={self.parent_seq_id}, "
                f"output_token={self.output_token}, "
                f"logprobs={self.logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceOutput):
            raise NotImplementedError()
        equal = (self.parent_seq_id == other.parent_seq_id
                 and self.output_token == other.output_token)
        log_probs_equal = other.logprobs == self.logprobs
        return equal and log_probs_equal


class SequenceGroupOutput(ABC):
    """The base class for model outputs associated with a sequence group."""

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass