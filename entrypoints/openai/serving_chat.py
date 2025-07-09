#!/usr/bin/env python3
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
# SPDX-License-Identifier: Apache-2.0
"""
Adapted from
https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/serving_chat.py
"""
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Final, Optional, Union

from vllm.entrypoints.chat_utils import ConversationMessage
from vllm.entrypoints.openai.protocol import (
    ChatCompletionNamedToolChoiceParam, ChatCompletionRequest,
    ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse,
    DeltaFunctionCall, DeltaMessage, DeltaToolCall, PromptTokenUsageInfo,
    RequestResponseMetadata, UsageInfo)
from vllm.entrypoints.openai.tool_parsers import ToolParser
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


async def chat_completion_stream_generator(
    self,
    request: ChatCompletionRequest,
    result_generator: AsyncIterator[RequestOutput],
    request_id: str,
    model_name: str,
    conversation: list[ConversationMessage],
    tokenizer: AnyTokenizer,
    request_metadata: RequestResponseMetadata,
) -> AsyncGenerator[str, None]:
    created_time = int(time.time())
    chunk_object_type: Final = "chat.completion.chunk"
    first_iteration = True

    # Send response for each token for each request.n (index)
    num_choices = 1 if request.n is None else request.n
    previous_num_tokens = [0] * num_choices
    finish_reason_sent = [False] * num_choices
    num_prompt_tokens = 0
    num_cached_tokens = None

    if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
        tool_choice_function_name = request.tool_choice.function.name
    else:
        tool_choice_function_name = None

    # Determine whether tools are in use with "auto" tool choice
    tool_choice_auto = (not tool_choice_function_name and
                        self._should_stream_with_auto_tool_parsing(request))

    should_stream_with_reasoning_parsing = (
        self._should_stream_with_reasoning_parsing(request))

    all_previous_token_ids: Optional[list[list[int]]]
    function_name_returned: Optional[list[bool]] = None

    # Only one of these will be used, thus previous_texts and
    # all_previous_token_ids will not be used twice in the same iteration.
    if tool_choice_auto or should_stream_with_reasoning_parsing:
        # These are only required in "auto" tool choice case
        previous_texts = [""] * num_choices
        all_previous_token_ids = [[]] * num_choices
        # For reasoning parser and tool call all enabled
        added_content_delta_arr = [False] * num_choices
        reasoning_end_arr = [False] * num_choices
    elif request.tool_choice == "required":
        previous_texts = [""] * num_choices
        function_name_returned = [False] * num_choices
        all_previous_token_ids = None
    else:
        previous_texts, all_previous_token_ids = None, None

    try:
        # There is no need to check if the reasoning_parser is None
        # because the should_stream_with_reasoning_parsing check
        # already ensures that the reasoning_parser is not None.
        # but the pre-commit hook requires it.
        if should_stream_with_reasoning_parsing and \
            self.reasoning_parser is not None:
            reasoning_parser = self.reasoning_parser(tokenizer)
    except RuntimeError as e:
        logger.exception("Error in reasoning parser creation.")
        data = self.create_streaming_error_response(str(e))
        yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Prepare the tool parser if it's needed
    try:
        if tool_choice_auto and self.tool_parser:
            tool_parsers: list[Optional[ToolParser]] = [
                self.tool_parser(tokenizer)
            ] * num_choices
        else:
            tool_parsers = [None] * num_choices
    except Exception as e:
        logger.exception("Error in tool parser creation.")
        data = self.create_streaming_error_response(str(e))
        yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"
        return

    stream_options = request.stream_options
    if stream_options:
        include_usage = stream_options.include_usage
        include_continuous_usage = include_usage and \
                                    stream_options.continuous_usage_stats
    else:
        include_usage, include_continuous_usage = False, False

    try:
        async for res in result_generator:
            if res.prompt_token_ids is not None:
                num_prompt_tokens = len(res.prompt_token_ids)
                if res.encoder_prompt_token_ids is not None:
                    num_prompt_tokens += len(res.encoder_prompt_token_ids)

            # We need to do it here, because if there are exceptions in
            # the result_generator, it needs to be sent as the FIRST
            # response (by the try...catch).
            if first_iteration:
                num_cached_tokens = res.num_cached_tokens
                # Send first response for each request.n (index) with
                # the role
                role = self.get_chat_request_role(request)

                # NOTE num_choices defaults to 1 so this usually executes
                # once per request
                for i in range(num_choices):
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(
                            role=role,
                            content="",
                        ),
                        logprobs=None,
                        finish_reason=None)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name)

                    # if continuous usage stats are requested, add it
                    if include_continuous_usage:
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=0,
                            total_tokens=num_prompt_tokens)

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

                # Send response to echo the input portion of the
                # last message
                if request.echo:
                    last_msg_content: Union[str, list[dict[str, str]]] = ""
                    if conversation and "content" in conversation[
                            -1] and conversation[-1].get("role") == role:
                        last_msg_content = conversation[-1]["content"] or ""

                    if last_msg_content:
                        for i in range(num_choices):
                            choice_data = (ChatCompletionResponseStreamChoice(
                                index=i,
                                delta=DeltaMessage(content=last_msg_content),
                                logprobs=None,
                                finish_reason=None))
                            chunk = ChatCompletionStreamResponse(
                                id=request_id,
                                object=chunk_object_type,
                                created=created_time,
                                choices=[choice_data],
                                model=model_name)
                            if include_continuous_usage:
                                chunk.usage = UsageInfo(
                                    prompt_tokens=num_prompt_tokens,
                                    completion_tokens=0,
                                    total_tokens=num_prompt_tokens)

                            data = chunk.model_dump_json(exclude_unset=True)
                            yield f"data: {data}\n\n"
                first_iteration = False

            for output in res.outputs:
                i = output.index
                tool_parser = tool_parsers[i]

                if finish_reason_sent[i]:
                    continue

                if request.logprobs and request.top_logprobs is not None:
                    assert output.logprobs is not None, (
                        "Did not output logprobs")
                    logprobs = self._create_chat_logprobs(
                        token_ids=output.token_ids,
                        top_logprobs=output.logprobs,
                        tokenizer=tokenizer,
                        num_output_top_logprobs=request.top_logprobs,
                        return_as_token_id=request.return_tokens_as_token_ids,
                    )
                else:
                    logprobs = None

                delta_text = output.text

                if not delta_text and not output.token_ids and \
                    not previous_num_tokens[i]:
                    # Chunked prefill case, don't return empty chunks
                    continue

                delta_message: Optional[DeltaMessage]

                # just update previous_texts and previous_token_ids
                if tool_choice_auto or should_stream_with_reasoning_parsing:
                    assert previous_texts is not None
                    assert all_previous_token_ids is not None
                    previous_text = previous_texts[i]
                    previous_token_ids = all_previous_token_ids[i]
                    current_text = previous_text + delta_text
                    current_token_ids = previous_token_ids + list(
                        output.token_ids)

                # handle streaming deltas for tools with named tool_choice
                if tool_choice_function_name:
                    if (self.enable_reasoning
                            and not reasoning_parser.is_reasoning_end(
                                previous_token_ids)):
                        assert reasoning_parser is not None
                        delta_message = (reasoning_parser.
                                         extract_reasoning_content_streaming(
                                             previous_text,
                                             current_text,
                                             delta_text,
                                             previous_token_ids,
                                             current_token_ids,
                                             output.token_ids,
                                         ))
                        # When encountering think end id in delta_token_ids,
                        # process the `content`. Only keep 'content',
                        # remove 'reasoning_content'
                        if reasoning_parser.is_reasoning_end(
                                list(output.token_ids)):
                            if delta_message and delta_message.content:
                                # This need to be added to next `delta_text`
                                current_text = delta_message.content
                                delta_message.content = None
                            else:
                                current_text = ""
                    else:
                        # Just to add remaining `content`
                        if self.enable_reasoning:
                            delta_text = previous_text + delta_text
                            current_text = ""

                        delta_message = DeltaMessage(tool_calls=[
                            DeltaToolCall(function=DeltaFunctionCall(
                                name=tool_choice_function_name,
                                arguments=delta_text),
                                          index=i)
                        ])

                elif request.tool_choice == "required":
                    assert previous_texts is not None
                    assert function_name_returned is not None
                    previous_text = previous_texts[i]
                    current_text = previous_text + delta_text
                    fn_name_returned = function_name_returned[i]

                    delta_message, function_name_returned[i] = (
                        self.extract_tool_call_required_streaming(
                            previous_text=previous_text,
                            current_text=current_text,
                            delta_text=delta_text,
                            function_name_returned=fn_name_returned))

                    # update the previous values for the next iteration
                    previous_texts[i] = current_text

                # handle streaming deltas for tools with "auto" tool choice
                # and reasoning parser
                elif tool_choice_auto and self.enable_reasoning:
                    assert tool_parser is not None
                    assert reasoning_parser is not None
                    assert added_content_delta_arr is not None
                    assert reasoning_end_arr is not None
                    if not reasoning_end_arr[i]:
                        delta_message = (reasoning_parser.
                                         extract_reasoning_content_streaming(
                                             previous_text,
                                             current_text,
                                             delta_text,
                                             previous_token_ids,
                                             current_token_ids,
                                             output.token_ids,
                                         ))

                        # When encountering think end id in delta_token_ids,
                        # set reasoning status to end.
                        # Remove the text and token ids related
                        # to 'reasoning_content'.
                        if reasoning_parser.is_reasoning_end(
                                list(output.token_ids)):
                            reasoning_end_arr[i] = True
                            current_token_ids =  \
                                reasoning_parser.extract_content_ids(
                                    list(output.token_ids))
                            if delta_message and delta_message.content:
                                current_text = delta_message.content
                                delta_message.content = None
                            else:
                                current_text = ""

                    # handle tool calls only after reasoning is done,
                    else:
                        delta_token_ids = list(output.token_ids)
                        # First time to tool call,
                        # add the remaining text and token ids
                        # to delta from previous
                        if not added_content_delta_arr[i]:
                            added_content_delta_arr[i] = True
                            previous_text = ""
                            previous_token_ids = []
                            delta_text = current_text
                            delta_token_ids = current_token_ids

                        delta_message = (
                            tool_parser.extract_tool_calls_streaming(
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                previous_token_ids=previous_token_ids,
                                current_token_ids=current_token_ids,
                                delta_token_ids=delta_token_ids,
                                request=request))
                # when only tool calls
                elif tool_choice_auto:
                    assert tool_parser is not None
                    delta_message = (tool_parser.extract_tool_calls_streaming(
                        previous_text=previous_text,
                        current_text=current_text,
                        delta_text=delta_text,
                        previous_token_ids=previous_token_ids,
                        current_token_ids=current_token_ids,
                        delta_token_ids=output.token_ids,
                        request=request))
                # when only reasoning
                elif self.enable_reasoning:
                    assert reasoning_parser is not None
                    delta_message = (
                        reasoning_parser.extract_reasoning_content_streaming(
                            previous_text,
                            current_text,
                            delta_text,
                            previous_token_ids,
                            current_token_ids,
                            output.token_ids,
                        ))
                # handle streaming just a content delta
                else:
                    delta_message = DeltaMessage(content=delta_text)

                # update the previous values for the next iteration
                if tool_choice_auto or should_stream_with_reasoning_parsing:
                    assert previous_texts is not None
                    assert all_previous_token_ids is not None
                    previous_texts[i] = current_text
                    all_previous_token_ids[i] = current_token_ids

                # set the previous values for the next iteration
                previous_num_tokens[i] += len(output.token_ids)

                # if the message delta is None (e.g. because it was a
                # "control token" for tool calls or the parser otherwise
                # wasn't ready to send a token, then
                #   get the next token without streaming a chunk
                if delta_message is None:
                    continue

                if output.finish_reason is None:
                    # Send token-by-token response for each request.n
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=delta_message,
                        logprobs=logprobs,
                        finish_reason=None)

                # if the model is finished generating
                else:
                    # check to make sure we haven't "forgotten" to stream
                    #   any tokens that were generated but previously
                    #   matched by partial json parsing
                    # only happens if we are NOT using guided decoding
                    auto_tools_called = False
                    if tool_parser:
                        auto_tools_called = len(
                            tool_parser.prev_tool_call_arr) > 0

                    # Send the finish response for each request.n only once
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=delta_message,
                        logprobs=logprobs,
                        finish_reason=output.finish_reason
                        if not auto_tools_called else "tool_calls",
                        stop_reason=output.stop_reason)

                    finish_reason_sent[i] = True

                chunk = ChatCompletionStreamResponse(id=request_id,
                                                     object=chunk_object_type,
                                                     created=created_time,
                                                     choices=[choice_data],
                                                     model=model_name)

                # handle usage stats if requested & if continuous
                if include_continuous_usage:
                    completion_tokens = previous_num_tokens[i]
                    chunk.usage = UsageInfo(
                        prompt_tokens=num_prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=num_prompt_tokens + completion_tokens,
                    )

                data = chunk.model_dump_json(exclude_unset=True)
                yield f"data: {data}\n\n"

        # once the final token is handled, if stream_options.include_usage
        # is sent, send the usage
        if include_usage:
            completion_tokens = sum(previous_num_tokens)
            final_usage = UsageInfo(prompt_tokens=num_prompt_tokens,
                                    completion_tokens=completion_tokens,
                                    total_tokens=num_prompt_tokens +
                                    completion_tokens)
            if self.enable_prompt_tokens_details and num_cached_tokens:
                final_usage.prompt_tokens_details = PromptTokenUsageInfo(
                    cached_tokens=num_cached_tokens)

            final_usage_chunk = ChatCompletionStreamResponse(
                id=request_id,
                object=chunk_object_type,
                created=created_time,
                choices=[],
                model=model_name,
                usage=final_usage)
            final_usage_data = (final_usage_chunk.model_dump_json(
                exclude_unset=True, exclude_none=True))
            yield f"data: {final_usage_data}\n\n"

        # report to FastAPI middleware aggregate usage across all choices
        num_completion_tokens = sum(previous_num_tokens)
        request_metadata.final_usage_info = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_completion_tokens,
            total_tokens=num_prompt_tokens + num_completion_tokens)

    except Exception as e:
        # TODO: Use a vllm-specific Validation Error
        logger.exception("Error in chat completion stream generator.")
        data = self.create_streaming_error_response(str(e))
        yield f"data: {data}\n\n"
    # Send the final done message after all response.n are finished
    yield "data: [DONE]\n\n"
