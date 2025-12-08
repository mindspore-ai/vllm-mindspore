# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/v1/core/sched/scheduler.py
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
"""Enhance system availability when error occurs in model-execution."""

# noqa: G004

import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Union

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVEventBatch
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                              create_request_queue)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs, FinishReason)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


# Refer to
# https://github.com/vllm-project/vllm/blob/main/vllm/
# v1/core/sched/scheduler.py
# Based on v0.11.0 branch commit ID: b8b302cde434df8c9289a2b465406b47ebab1c2d
class EnhancedScheduler(Scheduler):
    """
    Enhance the V1 scheduler with prefill-first scheduling strategy.
    This scheduler implements a prefill-first scheduling policy when chunked
    prefill is disabled, prioritizing prefill operations over decode operations
    to optimize performance for certain workloads.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(vllm_config, kv_cache_config,
                         structured_output_manager, mm_registry,
                         include_finished_set, log_stats)
        self.scheduled_req_ids: set[str] = set()
        self.ignored_req_ids: set[str] = set()

        self.max_kvcache_size = (self.kv_cache_config.num_blocks *
                                 self.block_size)
        self.model_runner_type = vllm_config.model_config.runner_type
        # add for type check
        self.running: list[Request] = []

    def schedule(self) -> SchedulerOutput:
        # Supports no chunked_prefill except for the pooling model
        if (self.scheduler_config.enable_chunked_prefill is False
                and self.model_runner_type != "pooling"):
            return self._schedule_no_chunked()

        # default
        return super().schedule()

    def _schedule_no_chunked(self) -> SchedulerOutput:
        """
        Schedule requests when chunked prefill is disabled.
        Implements a prefill-first scheduling strategy that:
        1. First schedules waiting prefill requests
        2. Then schedules running decode requests if token budget allows
        """
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # Record scheduled LoRA requests.
        scheduled_loras: set[int] = set()

        # For logging.
        scheduled_timestamp = time.monotonic()

        # vllm-mindspore: First, schedule the WAITING prefill requests.
        skipped_waiting_requests = create_request_queue(self.policy)
        req_index = 0
        while self.waiting and token_budget > 0:
            if len(self.running) == self.max_num_running_reqs:
                break

            request = self.waiting.peek_request()

            # KVTransfer: skip request if still waiting for remote kvs.
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                is_ready = self._update_waiting_for_remote_kv(request)
                if is_ready:
                    request.status = RequestStatus.WAITING
                else:
                    logger.debug(
                        "%s is still in WAITING_FOR_REMOTE_KVS state.",
                        request.request_id)
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

            # Skip request if the structured output request is still waiting
            # for FSM compilation.
            if request.status == RequestStatus.WAITING_FOR_FSM:
                structured_output_req = request.structured_output_request
                if structured_output_req and structured_output_req.grammar:
                    request.status = RequestStatus.WAITING
                else:
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

            # Check that adding the request still respects the max_loras
            # constraint.
            if self.lora_config and request.lora_request and (
                    len(scheduled_loras) == self.lora_config.max_loras and
                    request.lora_request.lora_int_id not in scheduled_loras):
                # Scheduling would exceed max_loras, skip.
                self.waiting.pop_request()
                skipped_waiting_requests.prepend_request(request)
                continue

            num_external_computed_tokens = 0
            load_kv_async = False

            # Get already-cached tokens.
            if request.num_computed_tokens == 0:
                new_computed_blocks, num_new_local_computed_tokens = \
                    self.kv_cache_manager.get_computed_blocks(
                        request)

                # Get externally-cached tokens if using a KVConnector.
                if self.connector is not None:
                    num_external_computed_tokens, load_kv_async = (
                        self.connector.get_num_new_matched_tokens(
                            request, num_new_local_computed_tokens))

                    if num_external_computed_tokens is None:
                        # The request cannot be scheduled because
                        # the KVConnector couldn't determine
                        # the number of matched tokens.
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Total computed tokens (local + external).
                num_computed_tokens = (num_new_local_computed_tokens +
                                       num_external_computed_tokens)
            # KVTransfer: WAITING reqs have num_computed_tokens > 0
            # after async KV recvs are completed.
            else:
                new_computed_blocks = (
                    self.kv_cache_manager.create_empty_block_list())
                num_new_local_computed_tokens = 0
                num_computed_tokens = request.num_computed_tokens

            encoder_inputs_to_schedule = None
            new_encoder_compute_budget = encoder_compute_budget

            # KVTransfer: loading remote KV, do not allocate for new work.
            if load_kv_async:
                assert num_external_computed_tokens > 0
                num_new_tokens = 0
            # Number of tokens to be scheduled.
            else:
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed
                # requests, which have output tokens.
                num_new_tokens = request.num_tokens - num_computed_tokens

                # vllm-mindspore: don't check the `chunked_prefill_enabled`
                # and `long_prefill_token_threshold`, if prompt length exceeds
                # the system limit, abort it.
                assert num_new_tokens >= 0
                prompt_limit = self._get_prompt_limit(request)
                prompt_limit = min(prompt_limit, self.max_kvcache_size)
                if num_new_tokens > prompt_limit:
                    logger.warning(
                        "Request(%s) prompt has (%d tokens) , and it \
                         exceeds system limit of %d. It will be ignored/abort.",
                        request.request_id,
                        num_new_tokens,
                        prompt_limit,
                    )
                    request.status = RequestStatus.FINISHED_IGNORED
                    self.ignored_req_ids.add(request.request_id)
                    self.waiting.pop_request()
                    continue

                # vllm-mindspore: skip if new tokens exceed token_budget
                if num_new_tokens > token_budget:
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                # Schedule encoder inputs.
                if request.has_encoder_inputs:
                    (encoder_inputs_to_schedule, num_new_tokens,
                     new_encoder_compute_budget
                     ) = self._try_schedule_encoder_inputs(
                         request, num_computed_tokens, num_new_tokens,
                         encoder_compute_budget)
                    if num_new_tokens == 0:
                        # The request cannot be scheduled.
                        break

            # Handles an edge case when P/D Disaggregation
            # is used with Spec Decoding where an
            # extra block gets allocated which
            # creates a mismatch between the number
            # of local and remote blocks.
            effective_lookahead_tokens = (0 if request.num_computed_tokens == 0
                                          else self.num_lookahead_tokens)

            # Determine if we need to allocate cross-attention blocks.
            if self.is_encoder_decoder and request.has_encoder_inputs:
                # TODO(russellb): For Whisper, we know that the input is
                # always padded to the maximum length. If we support other
                # encoder-decoder models, this will need to be updated if we
                # want to only allocate what is needed.
                num_encoder_tokens =\
                    self.scheduler_config.max_num_encoder_input_tokens
            else:
                num_encoder_tokens = 0

            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens + num_external_computed_tokens,
                num_new_local_computed_tokens,
                new_computed_blocks,
                num_lookahead_tokens=effective_lookahead_tokens,
                delay_cache_blocks=load_kv_async,
                num_encoder_tokens=num_encoder_tokens,
            )

            if new_blocks is None:
                # The request cannot be scheduled.
                break

            # KVTransfer: the connector uses this info to determine
            # if a load is needed. Note that
            # This information is used to determine if a load is
            # needed for this request.
            if self.connector is not None:
                self.connector.update_state_after_alloc(
                    request,
                    new_computed_blocks + new_blocks,
                    num_external_computed_tokens,
                )

            # Request was already popped from self.waiting
            # unless it was re-added above due to new_blocks being None.
            request = self.waiting.pop_request()
            if load_kv_async:
                # If loading async, allocate memory and put request
                # into the WAITING_FOR_REMOTE_KV state.
                skipped_waiting_requests.prepend_request(request)
                request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                continue

            req_index += 1
            self.running.append(request)

            # vllm-mindspore: add the req to the scheduled_req_ids queue
            self.scheduled_req_ids.add(request.request_id)

            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED,
                                     scheduled_timestamp)
            if request.status == RequestStatus.WAITING:
                scheduled_new_reqs.append(request)
            elif request.status == RequestStatus.PREEMPTED:
                scheduled_resumed_reqs.append(request)
            else:
                raise RuntimeError(f"Invalid request status: {request.status}")

            if self.lora_config and request.lora_request:
                scheduled_loras.add(request.lora_request.lora_int_id)
            req_to_new_blocks[request.request_id] = (
                self.kv_cache_manager.get_blocks(request.request_id))
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed_tokens
            # Count the number of prefix cached tokens.
            if request.num_cached_tokens < 0:
                request.num_cached_tokens = num_computed_tokens
            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # vllm-mindspore: Second, schedule the RUNNING decode requests
        # if no prefill reqs.
        if len(self.scheduled_req_ids) == 0:
            req_index = 0
            token_budget = self.max_num_scheduled_tokens

            while req_index < len(self.running) and token_budget > 0:
                request = self.running[req_index]

                # vllm-mindspore: The request has been scheduled.
                if request.request_id in self.scheduled_req_ids:
                    req_index += 1
                    continue

                num_new_tokens = (request.num_tokens_with_spec +
                                  request.num_output_placeholders -
                                  request.num_computed_tokens)

                # vllm-mindspore: don't check long_prefill_token_threshold
                num_new_tokens = min(num_new_tokens, token_budget)

                # Make sure the input position does not exceed the max model
                # len.This is necessary when using spec decoding.
                num_new_tokens = min(
                    num_new_tokens,
                    self.max_model_len - 1 - request.num_computed_tokens)

                # Schedule encoder inputs.
                encoder_inputs_to_schedule = None
                new_encoder_compute_budget = encoder_compute_budget
                if request.has_encoder_inputs:
                    (encoder_inputs_to_schedule, num_new_tokens,
                     new_encoder_compute_budget
                     ) = self._try_schedule_encoder_inputs(
                         request, request.num_computed_tokens, num_new_tokens,
                         encoder_compute_budget)

                if num_new_tokens == 0:
                    req_index += 1
                    continue

                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens)
                    if new_blocks is None:
                        # The request cannot be scheduled.
                        # Preempt the lowest-priority request.
                        if self.policy == SchedulingPolicy.PRIORITY:
                            preempted_req = max(
                                self.running,
                                key=lambda r: (r.priority, r.arrival_time),
                            )
                            self.running.remove(preempted_req)
                            if preempted_req in scheduled_running_reqs:
                                scheduled_running_reqs.remove(preempted_req)
                        else:
                            preempted_req = self.running.pop()

                        self.kv_cache_manager.free(preempted_req)
                        self.encoder_cache_manager.free(preempted_req)
                        preempted_req.status = RequestStatus.PREEMPTED
                        preempted_req.num_computed_tokens = 0
                        if self.log_stats:
                            preempted_req.record_event(
                                EngineCoreEventType.PREEMPTED,
                                scheduled_timestamp)

                        self.waiting.prepend_request(preempted_req)
                        preempted_reqs.append(preempted_req)
                        if preempted_req == request:
                            # No more request to preempt.
                            can_schedule = False
                            break
                    else:
                        # The request can be scheduled.
                        can_schedule = True
                        break
                if not can_schedule:
                    break
                assert new_blocks is not None

                # vllm-mindspore: add the req to the scheduled_req_ids queue
                self.scheduled_req_ids.add(request.request_id)

                # Schedule the request.
                scheduled_running_reqs.append(request)
                req_to_new_blocks[request.request_id] = new_blocks
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                req_index += 1

                # Speculative decode related.
                if request.spec_token_ids:
                    num_scheduled_spec_tokens = (num_new_tokens +
                                                 request.num_computed_tokens -
                                                 request.num_tokens)
                    if num_scheduled_spec_tokens > 0:
                        # Trim spec_token_ids list to num_scheduled_spec_tokens.
                        del request.spec_token_ids[num_scheduled_spec_tokens:]
                        scheduled_spec_decode_tokens[request.request_id] = (
                            request.spec_token_ids)

                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget

            # Record the LoRAs in scheduled_running_reqs
            # scheduled_loras has been defined before.
            if self.lora_config:
                scheduled_loras = set(
                    req.lora_request.lora_int_id
                    for req in scheduled_running_reqs
                    if req.lora_request and req.lora_request.lora_int_id > 0)
                assert len(scheduled_loras) <= self.lora_config.max_loras

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) <= len(self.running))

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(
            self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids())
            for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_blocks,
        )
        scheduled_requests = (scheduled_new_reqs + scheduled_running_reqs +
                              scheduled_resumed_reqs)
        structured_output_request_ids, grammar_bitmask = (
            self.get_grammar_bitmask(scheduled_requests,
                                     scheduled_spec_decode_tokens))
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.
            get_freed_mm_hashes(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        self._update_after_schedule(scheduler_output)
        return scheduler_output

    # Refer to vllm\core\scheduler.py Scheduler._get_prompt_limit
    def _get_prompt_limit(self, seq_group) -> int:
        """
        Get the maximum allowed prompt length for a sequence group.

        Takes into account model configuration limits and LoRA-specific
        maximum lengths for long context fine-tuned models.
        """
        prompt_limit = min(
            self.scheduler_config.max_model_len,
            self.scheduler_config.max_num_batched_tokens,
        )

        # Model is fine tuned with long context. Return the fine tuned max_len.
        if seq_group.lora_request and seq_group.lora_request.long_lora_max_len:
            return seq_group.lora_request.long_lora_max_len
        else:
            return prompt_limit

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        if len(self.scheduled_req_ids) > 0:
            for req_id in request_ids:
                request = self.requests.get(req_id)
                if request is None:
                    # Invalid request ID.
                    continue
                if request.status == RequestStatus.RUNNING:
                    # Clear the scheduled reqs
                    self.scheduled_req_ids.discard(request.request_id)

        super().finish_requests(request_ids, finished_status)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        if len(self.scheduled_req_ids) > 0:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens

            for request in self.running:
                req_id = request.request_id
                num_tokens_scheduled = num_scheduled_tokens.get(req_id, 0)
                if num_tokens_scheduled == 0:
                    continue
                # Clear the scheduled reqs
                if req_id in self.scheduled_req_ids:
                    self.scheduled_req_ids.discard(req_id)

        return self._update_from_output(scheduler_output, model_runner_output)

    def _update_from_output(
        self,
        scheduler_output,
        model_runner_output,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats = None
        kv_connector_stats = (kv_connector_output.kv_connector_stats
                              if kv_connector_output else None)

        # Add by vllm-mindspore begin:
        running_req_ids = [req.request_id for req in self.running]
        # abort_req_ids used to keep track of failed requests
        # caused by model execution exception
        abort_req_ids: list[str] = []
        # Add by vllm-mindspore end.

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            # Add by vllm-mindspore begin:
            # None sampled_token_ids comes from exception model execution,
            # set them to abort list
            # to keep main scheduler task running right.
            if sampled_token_ids is None:
                logger.warning(
                    'Process aborted request %s from running requests %s',
                    req_id, running_req_ids)
                outputs[request.client_index].append(
                    EngineCoreOutput(request_id=req_id,
                                     new_token_ids=[],
                                     finish_reason=FinishReason.ABORT,
                                     stop_reason=request.stop_reason,
                                     events=request.take_events()))
                abort_req_ids.append(req_id)
                continue
            # Add by vllm-mindspore end.

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[
                req_index] if sampled_token_ids else []

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id))
            if scheduled_spec_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                request.num_computed_tokens -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted)

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids)

            # Stop checking for pooler models.
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]
                stopped = check_stop(request, self.max_model_len,
                                     pooler_output)

            if stopped:
                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None \
                and request.sampling_params.logprobs is not None and logprobs:
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and self.structured_output_manager.should_advance(
                    request):
                # should not be None if use_structured_output, we have
                # checked above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None \
                or kv_transfer_params:

                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    ))
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Add by vllm-mindspore begin:
        # make failed requests finished to make the server
        # can continue to process new request
        if len(abort_req_ids) > 0:
            logger.warning('Aborted requests are %s', abort_req_ids)
            self.finish_requests(abort_req_ids, RequestStatus.FINISHED_ABORTED)

        self._process_ignored_reqs(outputs)
        # Add by vllm-mindspore end.

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        # KV Connector: update state for finished KV Transfers.
        if model_runner_output.kv_connector_output:
            self._update_from_kv_xfer_finished(
                model_runner_output.kv_connector_output)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set)
            finished_req_ids.clear()

        if (stats := self.make_stats(spec_decoding_stats,
                                     kv_connector_stats)) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    def _process_ignored_reqs(self, outputs):
        if len(self.ignored_req_ids) == 0:
            return

        for req_id in self.ignored_req_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue
            outputs[request.client_index].append(
                EngineCoreOutput(request_id=req_id,
                                 new_token_ids=[],
                                 finish_reason=request.get_finished_reason(),
                                 stop_reason=request.stop_reason,
                                 events=request.take_events()))

            # Clear the req source
            self._free_request(request)
        self.ignored_req_ids.clear()
