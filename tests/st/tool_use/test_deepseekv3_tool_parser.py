# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologies Co., Ltd.
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
"""Test tool parser for deepseek v3."""

# isort: skip_file
import vllm_mindspore

from collections.abc import Iterable
from typing import Union
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager

from tests.utils.common_utils import teardown_function, setup_function


class StreamingToolReconstructor:

    def __init__(self, assert_one_tool_per_delta: bool = True):
        self.tool_calls: list[ToolCall] = []
        self.other_content: str = ""
        self._assert_one_tool_per_delta = assert_one_tool_per_delta

    def append_delta(self, delta: DeltaMessage):
        if delta.content is not None:
            self.other_content += delta.content
        else:
            assert delta.tool_calls, (
                "Streaming results should have either content or tool calls "
                "(or both)")
        if self._assert_one_tool_per_delta:
            # Note: This isn't strictly required by the API and may not be
            # possible to adhere to depending on the token space and number of
            # tokens per streamed response from the model, but it is required
            # by tool_use tests, so we enforce it here by default also.
            assert len(delta.tool_calls) < 2, (
                "Streaming should include only one tool call per update.")
        for call_delta in delta.tool_calls:
            assert call_delta.type is None or call_delta.type == "function", (
                "Streaming tool calls should only emit function calls. Got "
                f"{call_delta.type}")
            current_tool_call = self.tool_calls[
                call_delta.index] if call_delta.index < len(
                    self.tool_calls) else None
            if current_tool_call:
                assert (not call_delta.function.name), (
                    "Streaming tool calls should emit the full function name "
                    f"exactly once. Got {call_delta.function.name}")
                assert (not call_delta.id), (
                    "Streaming tool calls must emit function id only once. Got "
                    f"{call_delta.id}")
                assert (call_delta.index == len(self.tool_calls) - 1), (
                    f"Incorrect index for tool delta. Got {call_delta.index}, "
                    f"expected {len(self.tool_calls) - 1}")
                current_tool_call.function.arguments += (
                    call_delta.function.arguments)
            else:
                assert call_delta.id is not None, (
                    "Streaming tool calls must have an id on first appearance")
                assert call_delta.function.name is not None, (
                    "Streaming tool calls must have a function name on first "
                    "appearance")
                assert call_delta.index == len(self.tool_calls), (
                    f"Incorrect index for tool delta. Got {call_delta.index}, "
                    f"expected {len(self.tool_calls)}")
                self.tool_calls.append(
                    ToolCall(id=call_delta.id,
                             function=FunctionCall(
                                 name=call_delta.function.name,
                                 arguments=call_delta.function.arguments
                                 or "")))


def run_tool_extraction(
    tool_parser: ToolParser,
    model_output: str,
    request: Union[ChatCompletionRequest, None] = None,
    streaming: bool = False,
    assert_one_tool_per_delta: bool = True,
) -> tuple[Union[str, None], list[ToolCall]]:
    if streaming:
        reconstructor = run_tool_extraction_streaming(
            tool_parser,
            model_output,
            request,
            assert_one_tool_per_delta=assert_one_tool_per_delta)
        return reconstructor.other_content or None, reconstructor.tool_calls
    else:
        extracted = run_tool_extraction_nonstreaming(tool_parser, model_output,
                                                     request)
        assert extracted.tools_called == bool(extracted.tool_calls)
        return extracted.content, extracted.tool_calls


def run_tool_extraction_nonstreaming(
    tool_parser: ToolParser,
    model_output: str,
    request: Union[ChatCompletionRequest, None] = None
) -> ExtractedToolCallInformation:
    request = request or ChatCompletionRequest(messages=[], model="test-model")
    return tool_parser.extract_tool_calls(model_output, request)


def run_tool_extraction_streaming(
    tool_parser: ToolParser,
    model_deltas: Iterable[str],
    request: Union[ChatCompletionRequest, None] = None,
    assert_one_tool_per_delta: bool = True,
) -> StreamingToolReconstructor:
    request = request or ChatCompletionRequest(messages=[], model="test-model")
    reconstructor = StreamingToolReconstructor(
        assert_one_tool_per_delta=assert_one_tool_per_delta)
    previous_text = ""
    previous_tokens: list[int] = []
    for delta in model_deltas:
        token_delta = [
            tool_parser.vocab.get(token)
            for token in tool_parser.model_tokenizer.tokenize(delta)
            if token in tool_parser.vocab
        ]
        current_text = previous_text + delta
        current_tokens = previous_tokens + token_delta
        delta_message = tool_parser.extract_tool_calls_streaming(
            previous_text, current_text, delta, previous_tokens,
            current_tokens, token_delta, request)
        if delta_message is not None:
            reconstructor.append_delta(delta_message)
        previous_text = current_text
        previous_tokens = current_tokens
    return reconstructor


SIMPLE_PARAMETER_FUNCTION_OUTPUT = (
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function"
    "<｜tool▁sep｜>get_weather\n```json\n"
    "{\"city\": \"LA\", \"metric\": \"C\"}\n```"
    "<｜tool▁call▁end｜><｜tool▁calls▁end｜>")
SIMPLE_PARAMETER_FUNCTION_CALL = FunctionCall(
    name="get_weather",
    arguments='{"city": "LA", "metric": "C"}',
)
MORE_TYPES_FUNCTION_OUTPUT = (
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function"
    "<｜tool▁sep｜>register_user\n```json\n"
    "{\"name\": \"Doe\", \"age\": 9, "
    "\"address\": {\"city\": \"LA\", \"state\": \"CA\"},"
    " \"role\": null, \"passed_test\": true, "
    "\"aliases\": [\"John\", \"Johnny\"]}\n```"
    "<｜tool▁call▁end｜><｜tool▁calls▁end｜>")
MORE_TYPES_FUNCTION_CALL = FunctionCall(
    name="register_user",
    arguments='{"name": "Doe", '
    '"age": 9, '
    '"address": {"city": "LA", "state": "CA"}, '
    '"role": null, '
    '"passed_test": true, '
    '"aliases": ["John", "Johnny"]}',
)
PARAMETERLESS_FUNCTION_OUTPUT = (
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function"
    "<｜tool▁sep｜>get_weather\n```json\n"
    "{}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>")
PARAMETERLESS_FUNCTION_CALL = FunctionCall(
    name="get_weather",
    arguments='{}',
)
EMPTY_DICT_FUNCTION_OUTPUT = (
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function"
    "<｜tool▁sep｜>do_something_cool\n```json\n{\"additional_data\": {}}\n```"
    "<｜tool▁call▁end｜><｜tool▁calls▁end｜>")
EMPTY_DICT_FUNCTION_CALL = FunctionCall(
    name="do_something_cool",
    arguments='{"additional_data": {}}',
)
EMPTY_LIST_FUNCTION_OUTPUT = (
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function"
    "<｜tool▁sep｜>do_something_cool\n```json\n{\"additional_data\": []}\n```"
    "<｜tool▁call▁end｜><｜tool▁calls▁end｜>")
EMPTY_LIST_FUNCTION_CALL = FunctionCall(
    name="do_something_cool",
    arguments='{"additional_data": []}',
)


@pytest.mark.parametrize("streaming", [True, False])
def test_no_tool_call(streaming: bool):
    mock_tokenizer = MagicMock()
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("deepseek_v3")(
        mock_tokenizer)
    model_output = "How can I help you today?"

    content, tool_calls = run_tool_extraction(tool_parser,
                                              model_output,
                                              streaming=streaming)

    assert content == model_output
    assert len(tool_calls) == 0


TEST_CASES = [
    pytest.param(False,
                 SIMPLE_PARAMETER_FUNCTION_OUTPUT,
                 [SIMPLE_PARAMETER_FUNCTION_CALL],
                 id="simple_nonstreaming"),
    pytest.param(False,
                 MORE_TYPES_FUNCTION_OUTPUT, [MORE_TYPES_FUNCTION_CALL],
                 id="more_types_nonstreaming"),
    pytest.param(False,
                 PARAMETERLESS_FUNCTION_OUTPUT, [PARAMETERLESS_FUNCTION_CALL],
                 id="parameterless_nonstreaming"),
    pytest.param(False,
                 EMPTY_DICT_FUNCTION_OUTPUT, [EMPTY_DICT_FUNCTION_CALL],
                 id="empty_dict_nonstreaming"),
    pytest.param(False,
                 EMPTY_LIST_FUNCTION_OUTPUT, [EMPTY_LIST_FUNCTION_CALL],
                 id="empty_list_nonstreaming"),
]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("streaming, model_output, expected_tool_calls",
                         TEST_CASES)
def test_tool_call(streaming: bool, model_output: str,
                   expected_tool_calls: list[FunctionCall]):
    """
    Test Summary:
        Test function call in various scenarios with deepseek_v3 tool_parser.
    Expected Result:
        Successful execution
    """
    mock_tokenizer = MagicMock()
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("deepseek_v3")(
        mock_tokenizer)

    content, tool_calls = run_tool_extraction(tool_parser,
                                              model_output,
                                              streaming=streaming)
    assert len(tool_calls) == len(expected_tool_calls)
    for actual, expected in zip(tool_calls, expected_tool_calls):
        assert actual.type == "function"
        assert actual.function == expected
