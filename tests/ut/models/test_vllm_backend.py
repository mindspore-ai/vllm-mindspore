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

import pytest

from tests.utils.common_utils import (teardown_function, setup_function,
                                      MODEL_PATH, start_vllm_server,
                                      get_key_counter_from_log,
                                      stop_vllm_server, send_and_get_request)
from tests.utils.env_var_manager import EnvVarManager

env_manager = EnvVarManager()
env_manager.setup_mindformers_environment()

QWEN_7B_MODEL = MODEL_PATH["Qwen2.5-7B-Instruct"]


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_vllm_backend_server_001():
    """
    Test Summary:
        Start models supported by both mindformers and native backends
        without explicitly specify VLLM_MS_MODEL_BACKEND, and perform
        online inference.
    Expected Result:
        Successful execution, executes under the minformers backend.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_backend_server_001.log"
    model = QWEN_7B_MODEL
    process = start_vllm_server(model, log_name)
    data = {
        "model":
        model,
        "messages": [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "aaaa",
            "content": "Who won the world series in 2020?"
        }],
        "max_tokens":
        100,
        "temperature":
        0
    }
    response = send_and_get_request(data, fmt="chat")
    stop_vllm_server(process)
    assert response.status_code == 200, response.text

    assert len(response.json()["choices"][0]["message"]["content"]) >= 100
    assert response.json()["choices"][0]["message"]["content"] == \
           "Sure, I'd be happy to help with any questions or topics you " \
           "have. Could you please provide more details about what you need " \
           "assistance with?" or response.json()[
               "choices"][0]["message"]["content"] == \
           "Sure, I'd be happy to help with any questions or topics you " \
           "have in mind. Could you please provide more details about what " \
           "you need assistance with?"
    result = get_key_counter_from_log(log_name, "mindformers")
    assert result > 5
    result = get_key_counter_from_log(log_name, "MindFormers backend")
    assert result >= 1
    result = get_key_counter_from_log(log_name,
                                      "Run with auto select model backend")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_vllm_backend_offline_001():
    """
    Test Summary:
        Start models supported by both mindformers and native backends
        without explicitly specify VLLM_MS_MODEL_BACKEND, and perform
        offline inference.
    Expected Result:
        Successful execution, executes under the minformers backend.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    import vllm_mindspore
    from vllm import LLM
    prompts = "Today is"
    sampling_params = None
    llm = LLM(model=QWEN_7B_MODEL)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        assert prompt == prompts
        assert output.finished is True
