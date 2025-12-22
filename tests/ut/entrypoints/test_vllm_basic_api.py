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
import os
import sys
import json
import requests
from unittest.mock import patch

from tests.utils.common_utils import (teardown_function, setup_function,
                                      MODEL_PATH, start_vllm_server,
                                      get_key_counter_from_log,
                                      stop_vllm_server, send_and_get_request)
from tests.utils.env_var_manager import EnvVarManager

import vllm_mindspore
from vllm import LLM, SamplingParams
from openai import OpenAI

env_manager = EnvVarManager()
env_manager.setup_mindformers_environment()

# def env
env_vars = {
    "VLLM_MS_MODEL_BACKEND": "Native",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
}

QWEN_7B_MODEL = MODEL_PATH["Qwen2.5-7B-Instruct"]
QWEN_32B_MODEL = MODEL_PATH["Qwen2.5-32B-Instruct"]


def run_vllm_offline_001():
    prompts = "Today is"
    sampling_params = None
    llm = LLM(model=QWEN_7B_MODEL)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        assert prompt == prompts
        assert output.finished


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_offline_001():
    """
    Test Summary:
        For the offline native backend scenario, the minimal configuration
        is used, prompts are passed as strings, and sampling_params is set
        to None.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    run_vllm_offline_001()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, {**env_vars, "VLLM_MS_MODEL_BACKEND": "MindFormers"})
def test_vllm_mf_offline_001():
    """
    Test Summary:
        For the offline mindformers backend scenario, the minimal configuration
        is used, prompts are passed as strings, and sampling_params is set
        to None.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    run_vllm_offline_001()


def run_vllm_offline_002(model, tensor_parallel_size=1):
    prompts = "Today is"
    sampling_params = SamplingParams(n=3,
                                     top_k=3,
                                     top_p=0.5,
                                     temperature=2.0,
                                     repetition_penalty=2.0)
    llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        assert prompt == prompts
        assert output.finished
        assert len(output.outputs) == 3


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_offline_002():
    """
    Test Summary:
        For the offline native backend scenario, test the 4 post-processing
        parameters supported by the sampling_params configuration.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    run_vllm_offline_002(model=QWEN_7B_MODEL)


@pytest.mark.level1
@patch.dict(os.environ, {**env_vars, "VLLM_MS_MODEL_BACKEND": "MindFormers"})
def test_vllm_mf_offline_002():
    """
    Test Summary:
        For the offline mindformers backend scenario, test the 4
        post-processing parameters supported by the sampling_params
        configuration.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-32B-Instruct
    """
    run_vllm_offline_002(model=QWEN_32B_MODEL, tensor_parallel_size=2)


def run_vllm_offline_003():
    prompts = ""
    sampling_params = SamplingParams(top_k=1)
    llm = LLM(model=QWEN_7B_MODEL)
    with pytest.raises(ValueError) as err:
        llm.generate(prompts, sampling_params)
    assert "The decoder prompt cannot be empty" in str(err.value)
    prompts = ["", "Today is", "Llama is"]
    with pytest.raises(ValueError) as err:
        llm.generate(prompts, sampling_params)
    assert "The decoder prompt cannot be empty" in str(err.value)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_offline_003():
    """
    Test Summary:
        For the offline native backend scenario, test the prompts parameter
        in llm.generate is an empty string.
    Expected Result:
        Execution fails with a ValueError.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    run_vllm_offline_003()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, {**env_vars, "VLLM_MS_MODEL_BACKEND": "MindFormers"})
def test_vllm_mf_offline_003():
    """
    Test Summary:
        For the offline mindformers backend scenario, test the prompts
        parameter in llm.generate is an empty string.
    Expected Result:
        Execution fails with a ValueError.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    run_vllm_offline_003()


def run_vllm_offline_004():
    prompts = ["I am", "Today is", "I love Beijing, because"]
    sampling_params = SamplingParams(temperature=0.0, logprobs=4)
    llm = LLM(model=QWEN_7B_MODEL)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        assert output.finished
        for i in range(len(output.outputs[0].token_ids)):
            assert len(output.outputs[0].logprobs[i]) >= 4
    assert outputs[2].outputs[0].text == \
           " it is a city with a long history. " + \
           "Which of the following options correctly expresses"


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_offline_004():
    """
    Test Summary:
        For the offline native backend scenario, test the prompts parameter
        in llm.generate is a list of strings (list[str]).
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    run_vllm_offline_004()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, {**env_vars, "VLLM_MS_MODEL_BACKEND": "MindFormers"})
def test_vllm_mf_offline_004():
    """
    Test Summary:
        For the offline mindformers backend scenario, test the prompts
        parameter in llm.generate is a list of strings (list[str]).
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    run_vllm_offline_004()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_server_001():
    """
    Test Summary:
        For the online native backend scenario, test request interface,
        using the minimal configuration with prompts passed as strings.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_server_001.log"
    model = QWEN_7B_MODEL
    process = start_vllm_server(model, log_name)
    openai_api_key = "EMPTY"
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        openai_api_base = f'http://localhost:{serve_port}/v1'
    else:
        openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    _ = client.completions.create(model=model, prompt="Today is")
    stop_vllm_server(process)
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1


@pytest.mark.level0
@patch.dict(os.environ, {**env_vars, "VLLM_MS_MODEL_BACKEND": "MindFormers"})
def test_vllm_mf_server_001():
    """
    Test Summary:
        For the online mindformers backend scenario, test request interface,
        using the minimal configuration with prompts passed as strings.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-32B-Instruct
    """
    log_name = "test_vllm_mf_server_001.log"
    model = QWEN_32B_MODEL
    process = start_vllm_server(model,
                                log_name,
                                extra_params='--tensor_parallel_size=2 ')
    openai_api_key = "EMPTY"
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        openai_api_base = f'http://localhost:{serve_port}/v1'
    else:
        openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    models = client.models.list()
    model = models.data[0].id
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": "Who won the world series in 2020?"
        }, {
            "role":
            "assistant",
            "content":
            "The Los Angeles Dodgers won the World Series in 2020."
        }, {
            "role": "user",
            "content": "Where was it played?"
        }],
        model=model,
    )
    stop_vllm_server(process)
    assert chat_completion.choices[0].finish_reason == 'stop'
    result = get_key_counter_from_log(log_name, "Run with Mindformers backend")
    assert result >= 1


def run_vllm_server_002(log_name):
    model = QWEN_7B_MODEL
    process = start_vllm_server(model, log_name)
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        url = f'http://localhost:{serve_port}/v1/completions'
    else:
        url = 'http://localhost:8000/v1/completions'
    data = {
        "model": model,
        "prompt": ["I am", "Today is", "Llama is"],
        "top_k": 1
    }
    json_data = json.dumps(data)
    response = requests.post(url,
                             data=json_data,
                             headers={'Content-Type': 'application/json'})
    stop_vllm_server(process)
    assert response.status_code == 200


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_server_002():
    """
    Test Summary:
         For the online native backend scenario, test request interface
         with prompts as a list of strings (list[str]).
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_server_002.log"
    run_vllm_server_002(log_name)
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, {**env_vars, "VLLM_MS_MODEL_BACKEND": "MindFormers"})
def test_vllm_mf_server_002():
    """
    Test Summary:
         For the online mindformers backend scenario, test request interface
         with prompts as a list of strings (list[str]).
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_mf_server_002.log"
    run_vllm_server_002(log_name)
    result = get_key_counter_from_log(log_name, "Run with Mindformers backend")
    assert result >= 1


def run_vllm_server_003(log_name):
    model = QWEN_32B_MODEL
    extra_params = "--tensor_parallel_size=2"
    process = start_vllm_server(model, log_name, extra_params=extra_params)
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        url = f'http://localhost:{serve_port}/v1/completions'
    else:
        url = 'http://localhost:8000/v1/completions'
    data = {
        "model": model,
        "prompt": ["I am", "Today is", "Llama is"],
        "repetition_penalty": 0.5
    }
    json_data = json.dumps(data)
    response1 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["repetition_penalty"] = 1.5
    json_data = json.dumps(data)
    response2 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["repetition_penalty"] = 2
    json_data = json.dumps(data)
    response3 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["repetition_penalty"] = 2.5
    json_data = json.dumps(data)
    response4 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data = {
        "model": model,
        "prompt": "I love Beijing, because",
        "temperature": 0
    }
    json_data = json.dumps(data)
    response5 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    stop_vllm_server(process)
    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response3.status_code == 200
    assert response4.status_code == 200
    assert " it is the capital of China." in response5.json(
    )["choices"][0]["text"]


@pytest.mark.level1
@patch.dict(os.environ, env_vars)
def test_vllm_server_003():
    """
    Test Summary:
        For the online native backend scenario, test request interface with
        post-processing parameter combination:
        temperature, top_k, and top_p use default values.
        repetition_penalty is configured as 0.5, 1.5, 2, and 2.5.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-32B-Instruct
    """
    log_name = "test_vllm_server_003.log"
    run_vllm_server_003(log_name)
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1


@pytest.mark.level1
@patch.dict(os.environ, {**env_vars, "VLLM_MS_MODEL_BACKEND": "MindFormers"})
def test_vllm_mf_server_003():
    """
    Test Summary:
        For the online mindformers backend scenario, test request interface
        with post-processing parameter combination:
        temperature, top_k, and top_p use default values.
        repetition_penalty is configured as 0.5, 1.5, 2, and 2.5.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-32B-Instruct
    """
    log_name = "test_vllm_mf_server_003.log"
    run_vllm_server_003(log_name)
    result = get_key_counter_from_log(log_name, "Run with Mindformers backend")
    assert result >= 1


def run_vllm_server_004(log_name):
    model = QWEN_7B_MODEL
    process = start_vllm_server(model, log_name)
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        url = f'http://localhost:{serve_port}/v1/completions'
    else:
        url = 'http://localhost:8000/v1/completions'
    data = {"model": model, "prompt": [15364, 374], "temperature": 0}
    json_data = json.dumps(data)
    response1 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["temperature"] = 0.0001
    json_data = json.dumps(data)
    response2 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["temperature"] = 0.001
    json_data = json.dumps(data)
    response3 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["temperature"] = 2
    json_data = json.dumps(data)
    response4 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})

    stop_vllm_server(process)
    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response3.status_code == 200
    assert response4.status_code == 200


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_server_004():
    """
    Test Summary:
        For the online native backend scenario, test request interface
        with post-processing parameter combination:
        repetition_penalty, top_k, and top_p use default values.
        temperature is configured as 0, 0.0001, 0.001, and 2.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_server_004.log"
    run_vllm_server_004(log_name)
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, {**env_vars, "VLLM_MS_MODEL_BACKEND": "MindFormers"})
def test_vllm_mf_server_004():
    """
    Test Summary:
        For the online mindformers backend scenario, test request interface
        with post-processing parameter combination:
        repetition_penalty, top_k, and top_p use default values.
        temperature is configured as 0, 0.0001, 0.001, and 2.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_mf_server_004.log"
    run_vllm_server_004(log_name)
    result = get_key_counter_from_log(log_name, "Run with Mindformers backend")
    assert result >= 1


def run_vllm_server_005(log_name):
    model = QWEN_7B_MODEL
    with open(os.path.join(model, "tokenizer.json")) as f:
        tokens = json.load(f)
    vocab_len = len(tokens['model']['vocab'])
    process = start_vllm_server(model, log_name)
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        url = f'http://localhost:{serve_port}/v1/completions'
    else:
        url = 'http://localhost:8000/v1/completions'
    data = {"model": model, "prompt": [[15364, 374], [15364, 374]], "top_k": 0}

    json_data = json.dumps(data)
    response1 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})

    # Far larger than the size of the vocab_len
    data["top_k"] = sys.maxsize
    json_data = json.dumps(data)
    response2 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["top_k"] = vocab_len
    json_data = json.dumps(data)
    response3 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["top_k"] = vocab_len - 1
    json_data = json.dumps(data)
    response4 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})

    stop_vllm_server(process)
    assert response1.status_code == 200
    assert response2.status_code == 400
    assert response3.status_code == 200
    assert response4.status_code == 200


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_server_005():
    """
    Test Summary:
        For the online native backend scenario, test request interface
        with post-processing parameter combination:
        repetition_penalty, temperature, and top_p use default values.
        top_k covers 0, maximum integer, vocabulary size, and vocabulary
        size - 1.
    Expected Result:
        Except for the scenario where the value exceeds the vocabulary size,
        all other scenarios execute successfully with normal inference results.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_server_005.log"
    run_vllm_server_005(log_name)
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, {**env_vars, "VLLM_MS_MODEL_BACKEND": "MindFormers"})
def test_vllm_mf_server_005():
    """
    Test Summary:
        For the online mindformers backend scenario, test request interface
        with post-processing parameter combination:
        repetition_penalty, temperature, and top_p use default values.
        top_k covers 0, maximum integer, vocabulary size, and vocabulary
        size - 1.
    Expected Result:
        Except for the scenario where the value exceeds the vocabulary size,
        all other scenarios execute successfully with normal inference results.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_mf_server_005.log"
    run_vllm_server_005(log_name)
    result = get_key_counter_from_log(log_name, "Run with Mindformers backend")
    assert result >= 1


def run_vllm_server_006(log_name):
    model = QWEN_7B_MODEL
    process = start_vllm_server(model, log_name)
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        url = f'http://localhost:{serve_port}/v1/completions'
    else:
        url = 'http://localhost:8000/v1/completions'
    data = {"model": model, "prompt": "I am", "top_p": 0}
    json_data = json.dumps(data)
    response1 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["top_p"] = 0.3
    json_data = json.dumps(data)
    response2 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["top_p"] = 0.5
    json_data = json.dumps(data)
    response3 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["top_p"] = 0.8
    json_data = json.dumps(data)
    response4 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})

    stop_vllm_server(process)
    assert response1.status_code == 400
    assert response2.status_code == 200
    assert response3.status_code == 200
    assert response4.status_code == 200


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_server_006():
    """
    Test Summary:
        For the online native backend scenario, test request interface
        with post-processing parameter combination:
        repetition_penalty, temperature, and top_k use default values.
        top_p covers 0, 0.3, 0.5, and 0.8.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_server_006.log"
    run_vllm_server_006(log_name)
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, {**env_vars, "VLLM_MS_MODEL_BACKEND": "MindFormers"})
def test_vllm_mf_server_006():
    """
    Test Summary:
        For the online mindformers backend scenario, test request interface
        with post-processing parameter combination:
        repetition_penalty, temperature, and top_k use default values.
        top_p covers 0, 0.3, 0.5, and 0.8.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_mf_server_006.log"
    run_vllm_server_006(log_name)
    result = get_key_counter_from_log(log_name, "Run with Mindformers backend")
    assert result >= 1


def run_vllm_server_007(log_name):
    model = QWEN_7B_MODEL
    with open(os.path.join(model, "tokenizer.json")) as f:
        tokens = json.load(f)
    vocab_len = len(tokens['model']['vocab'])
    process = start_vllm_server(model, log_name)
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        url = f'http://localhost:{serve_port}/v1/completions'
    else:
        url = 'http://localhost:8000/v1/completions'
    data = {
        "model": model,
        "prompt": "I am",
        "repetition_penalty": 1.5,
        "temperature": 0.001,
        "top_k": 5,
        "top_p": 0.5
    }
    json_data = json.dumps(data)
    response1 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data = {
        "model": model,
        "prompt": "I am",
        "repetition_penalty": 1,
        "temperature": 2,
        "top_k": vocab_len - 1,
        "top_p": 1
    }
    json_data = json.dumps(data)
    response2 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data = {
        "model": model,
        "prompt": "I am",
        "repetition_penalty": 2,
        "temperature": 0.001,
        "top_k": 1,
        "top_p": 1
    }
    json_data = json.dumps(data)
    response3 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    stop_vllm_server(process)
    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response3.status_code == 200


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_server_007():
    """
    Test Summary:
        For the online native backend scenario, test request interface
        with post-processing parameter combination:
        repetition_penalty=1.5 temperature=0.001 top_k=5 top_p=0.5
        repetition_penalty=1 temperature=2 top_k=vocabSize-1 top_p=1
        repetition_penalty=2 temperature=0.001 top_k=1 top_p=1
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_server_007.log"
    run_vllm_server_007(log_name)
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, {**env_vars, "VLLM_MS_MODEL_BACKEND": "MindFormers"})
def test_vllm_mf_server_007():
    """
    Test Summary:
        For the online mindformers backend scenario, test request interface
        with post-processing parameter combination:
        repetition_penalty=1.5 temperature=0.001  top_k=5 top_p=0.5
        repetition_penalty=1 temperature=2  top_k=vocabSize-1 top_p=1
        repetition_penalty=2 temperature=0.001 top_k=1 top_p=1
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_mf_server_007.log"
    run_vllm_server_007(log_name)
    result = get_key_counter_from_log(log_name, "Run with Mindformers backend")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_offline_err_001():
    """
    Test Summary:
        Test abnormal parameter values for SamplingParams in the native
        backend offline scenario.
    Expected Result:
        Raises ValueError, and the corresponding error message is verified.
    """
    with pytest.raises(ValueError) as err:
        SamplingParams(top_k=-5)
    assert "top_k must be 0 (disable), or at least 1, got -5." in str(
        err.value)
    with pytest.raises(ValueError) as err:
        SamplingParams(top_p=0)
    assert "top_p must be in (0, 1], got 0." in str(err.value)
    with pytest.raises(ValueError) as err:
        SamplingParams(temperature=-1)
    assert "temperature must be non-negative, got -1." in str(err.value)
    with pytest.raises(ValueError) as err:
        SamplingParams(repetition_penalty=-2.0)
    assert "repetition_penalty must be greater than zero, got -2.0." in str(
        err.value)
    SamplingParams(top_k=sys.maxsize)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_offline_err_002():
    """
    Test Summary:
        Abnormal parameter values for the LLM interface in the offline
        scenario.
    Expected Result:
        Raises ValueError and TypeError, and the corresponding error
        message is verified.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    with pytest.raises(TypeError) as err:
        LLM(model=1)
    assert "expected str, bytes or os.PathLike object, not int" in str(
        err.value)
    with pytest.raises(Exception) as err:
        LLM(model="/home/workspace/mindspore_dataset/weight/")
    assert "Repo id must be in the form 'repo_name' or " \
           "'namespace/repo_name'" in str(err.value) or \
           "Invalid repository ID or local directory " \
           "specified" in str(err.value)
    llm = LLM(model=QWEN_7B_MODEL)
    with pytest.raises(TypeError) as err:
        llm.generate(1, None)
    assert "object of type 'int' has no len()" in str(err.value)
    with pytest.raises(AttributeError) as err:
        llm.generate("i am", sampling_params=1)
    assert "'int' object has no attribute 'truncate_prompt_tokens'" in str(
        err.value)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_model_alias_ms_server_001():
    """
    Test Summary:
        For the online native backend scenario, The model path is configured
        as /home/path, with served-model-name. Configure 8 model names: name1,
        name2, name2 (name2 is duplicated), ..., name8 (covering a total of
        8 names). And then send requests using the model name "name2", send
        requests using the model name "name8".
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_v073_ms_server_001.log"
    model = QWEN_7B_MODEL
    process = start_vllm_server(
        model,
        log_name,
        extra_params="--served-model-name qwen1 qwen2 qwen2 qwen3 qwen4 "
        "qwen5 qwen6 qwen7 qwen8")
    data = {
        "model": model,
        "prompt": "I am",
        "max_tokens": 100,
        "temperature": 0
    }
    response = send_and_get_request(data)
    data["model"] = "qwen2"
    response1 = send_and_get_request(data)
    data["model"] = "qwen8"
    response2 = send_and_get_request(data)
    stop_vllm_server(process)
    assert response.status_code == 404, response.text
    assert response1.status_code == 200, response1.text
    assert response2.status_code == 200, response2.text
    assert response1.json()["choices"][0]["text"] == \
           response2.json()["choices"][0]["text"]
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, {**env_vars, "VLLM_MS_MODEL_BACKEND": "MindFormers"})
def test_vllm_mf_server_err_001():
    """
    Test Summary:
        For the online mindformers backend scenario, test abnormal parameter
        values for SamplingParams.
    Expected Result:
        Execution fails with the error message containing the keyword:
        "repetition_penalty must be greater than zero".
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_mf_server_err_001.log"
    model = QWEN_7B_MODEL
    process = start_vllm_server(model, log_name)
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        url = f'http://localhost:{serve_port}/v1/completions'
    else:
        url = 'http://localhost:8000/v1/completions'
    data = {
        "model": model,
        "prompt": ["", "Today is", "Llama is"],
        "top_k": -0.2
    }
    json_data = json.dumps(data)
    response1 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data = {"model": model, "prompt": ["", "Today is", "Llama is"], "top_p": 5}
    json_data = json.dumps(data)
    response2 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data = {
        "model": model,
        "prompt": ["", "Today is", "Llama is"],
        "temperature": -2
    }
    json_data = json.dumps(data)
    response3 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data = {
        "model": model,
        "prompt": ["", "Today is", "Llama is"],
        "repetition_penalty": -2
    }
    json_data = json.dumps(data)
    response4 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    stop_vllm_server(process)
    assert response1.status_code == 400
    assert "Input should be a valid integer, got a number with a fractional " \
           "part" in response1.json()['error']['message']
    assert response2.status_code == 400
    assert "top_p must be in (0, 1], got 5.0." in response2.json(
    )['error']['message']
    assert response3.status_code == 400
    assert "temperature must be non-negative, got -2.0." in \
           response3.json()['error']['message']
    assert response4.status_code == 400
    assert "repetition_penalty must be greater than zero, got -2.0." in \
           response4.json()['error']['message']


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, {**env_vars, "VLLM_MS_MODEL_BACKEND": "MindFormers"})
def test_vllm_mf_server_err_002():
    """
    Test Summary:
        For the online mindformers backend scenario, test abnormal parameter
        values of parameters other than SamplingParams.
    Expected Result:
        Execution fails with an error message indicating the correct
        format requirements.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_mf_server_err_002.log"
    model = QWEN_7B_MODEL
    process = start_vllm_server(model, log_name)
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        url = f'http://localhost:{serve_port}/v1/completions'
    else:
        url = 'http://localhost:8000/v1/completions'
    data = {"model": True, "prompt": ["", "Today is", "Llama is"]}
    json_data = json.dumps(data)
    response1 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data = {"model": "/var/log", "prompt": ["", "Today is", "Llama is"]}
    json_data = json.dumps(data)
    response2 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data = {"model": model, "prompt": 1}
    json_data = json.dumps(data)
    response3 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    stop_vllm_server(process)
    assert response1.status_code == 400
    assert "Input should be a valid string" in response1.json(
    )['error']['message']
    assert response2.status_code == 404
    assert "The model `/var/log` does not exist." in \
           response2.json()['error']['message']
    assert response3.status_code == 400
    assert "Input should be a valid list" in response3.json(
    )['error']['message']


@pytest.mark.level3
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, {**env_vars, "VLLM_MS_MODEL_BACKEND": "MindFormers"})
def test_vllm_v073_mf_server_002():
    """
    Test Summary:
        Error handling test for chat requests in the llm.chat online server
        scenario.
    Expected Result:
        Successful execution and the returned result contains prompt
        information.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_v073_mf_server_002.log"
    model = QWEN_7B_MODEL
    process = start_vllm_server(model, log_name)
    data = {
        "model":
        model,
        "messages": [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "anonymous",
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
           "have in mind. Could you please provide more details about what " \
           "you need assistance with?"
    result = get_key_counter_from_log(log_name, "Run with Mindformers backend")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, {**env_vars, "VLLM_MS_MODEL_BACKEND": "MindFormers"})
def test_vllm_v073_mf_server_005():
    """
    Test Summary:
        Based on the mindformers backend, deploy online server and perform
        inference process by accessing the /v1/chat/completions endpoint
        as follow:
        1.Send a single request via curl.
        2.Include requests with multi-turn conversations or requests from
        multiple types of users.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_v073_mf_server_005.log"
    model = QWEN_7B_MODEL
    process = start_vllm_server(model, log_name)
    data = {
        "model":
        model,
        "messages": [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": "Who won the world series in 2020?"
        }],
        "max_tokens":
        100,
        "temperature":
        0
    }
    response = send_and_get_request(data, fmt="chat")
    data = {
        "model":
        model,
        "messages": [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": "Hello!"
        }, {
            "role": "assistant",
            "content": "Hello! How can I help you?"
        }, {
            "role": "user",
            "content": "What is the capital of France?"
        }, {
            "role": "assistant",
            "content": "The capital of France is Paris."
        }, {
            "role": "user",
            "content": "What is the population of Paris?"
        }],
        "max_tokens":
        100,
        "temperature":
        0
    }
    response1 = send_and_get_request(data, fmt="chat")
    stop_vllm_server(process)
    assert response.status_code == 200, response.text
    assert response1.status_code == 200, response1.text
    assert "The Tampa Bay Rays won the World Series in 2020" \
            in response.json()["choices"][0]["message"]["content"]
    assert "As of 2023, the population of Paris is approximately 2.2" \
            in response1.json()["choices"][0]["message"]["content"]
    result = get_key_counter_from_log(log_name, "Run with Mindformers backend")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, {**env_vars, "VLLM_MS_MODEL_BACKEND": "MindFormers"})
def test_vllm_v073_mf_server_006():
    """
    Test Summary:
        Based on the mindformers backend, deploy online server with the model
        path of /home/path, and served-model-name is configured with name1,
        name2, ...
        And then:
        Send a request using the model name "/home/path".
        Send a request using the model name "name1".
        Send a request using the model name "name2".
        Send a request using the model name "name3".
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_v073_mf_server_006.log"
    model = QWEN_7B_MODEL
    process = start_vllm_server(
        model,
        log_name,
        extra_params="--served-model-name qwen0 qwen0 qwen1 qwen2 qwen3")
    data = {
        "model": model,
        "prompt": "I am",
        "max_tokens": 100,
        "temperature": 0
    }
    response = send_and_get_request(data)
    data["model"] = "qwen0"
    response1 = send_and_get_request(data)
    data["model"] = "qwen1"
    response2 = send_and_get_request(data)
    data["model"] = "qwen2"
    response3 = send_and_get_request(data)
    stop_vllm_server(process)
    assert response.status_code == 404, response.text
    assert response1.status_code == 200, response1.text
    assert response2.status_code == 200, response2.text
    assert response3.status_code == 200, response3.text
    assert response1.json()["choices"][0]["text"] == response2.json(
    )["choices"][0]["text"]
    assert response1.json()["choices"][0]["text"] == response3.json(
    )["choices"][0]["text"]
    result = get_key_counter_from_log(log_name, "Run with Mindformers backend")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, {**env_vars, "VLLM_MS_MODEL_BACKEND": "MindFormers"})
def test_vllm_v073_mf_offline_001():
    """
    Test Summary:
        For the offline mindformers backend scenario, testing configure
        arbitrary post-processing parameters, and send a long request (16k
        in length).
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    model = QWEN_7B_MODEL
    message = "Hello!" * 10000
    sampling_params = SamplingParams(top_k=3,
                                     top_p=0.5,
                                     temperature=0.0,
                                     repetition_penalty=2.0)
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": message
    }]
    llm = LLM(model=model)
    outputs = llm.chat(messages, sampling_params)
    for output in outputs:
        assert "Greetings and welcome to you, my friend!" in output.outputs[
            0].text, output.outputs[0].text
        assert output.finished is True


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, {**env_vars, "VLLM_MS_MODEL_BACKEND": "MindFormers"})
def test_vllm_v073_mf_offline_002():
    """
    Test Summary:
        For the offline mindformers backend scenario, test error handling
        for incoming requests in the llm.chat interface.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    model = QWEN_7B_MODEL
    message = "Hello!"
    sampling_params = SamplingParams(temperature=0.0)
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "anonymous",
        "content": message
    }]
    llm = LLM(model=model)
    outputs = llm.chat(messages, sampling_params)
    for output in outputs:
        assert output.finished is True
