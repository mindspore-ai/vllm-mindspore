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
import os
import json
import requests
import pytest
from unittest.mock import patch
from tests.utils.common_utils import (teardown_function, setup_function,
                                      MODEL_PATH, logger, start_vllm_server,
                                      get_key_counter_from_log,
                                      stop_vllm_server,
                                      run_combination_accuracy)

# def env
env_vars = {
    "VLLM_MS_MODEL_BACKEND": "Native",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
}

QWEN_7B_MODEL = MODEL_PATH["Qwen2.5-7B-Instruct"]

# Long prompts with more than max_num_batched_tokens tokens
LONG_PROMPT = "I love Beijing, because it is a city with a long history " \
              "and profound cultural heritage. Walking through its ancient " \
              "hutongs, one can almost feel the whispers of the past. The " \
              "Forbidden City, an architectural marvel that once housed " \
              "emperors, stands as a testament to the city's imperial " \
              "past. Meanwhile, the Great Wall, though not within the city " \
              "limits, is easily accessible from Beijing and offers a " \
              "glimpse into the strategic genius and resilience of ancient " \
              "China."


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_ms_offline_chunked_prefill_001():
    """
    Test Summary:
        For the offline native backend scenario with default mode,
        Test sending single batch request whose length exceeding the
        max_num_batched_tokens.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    import vllm_mindspore
    from vllm import LLM, SamplingParams
    prompts = LONG_PROMPT
    sampling_params = SamplingParams(temperature=0.0,
                                     top_p=0.95,
                                     max_tokens=100)
    llm = LLM(QWEN_7B_MODEL, max_num_seqs=16, max_num_batched_tokens=32)
    outputs = llm.generate(prompts, sampling_params)
    logger.info(outputs)
    for output in outputs:
        prompt = output.prompt
        assert prompt == prompts
        assert output.finished is True


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_ms_offline_chunked_prefill_003():
    """
    Test Summary:
        For the offline native backend scenario with default mode,
        Test sending multiple batch requests, where all requests do not
        exceed max_num_batched_tokens.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    import vllm_mindspore
    from vllm import LLM, SamplingParams
    prompts = ["I love Beijing", "Today is", "Llama is"]
    sampling_params = SamplingParams(temperature=0.0,
                                     top_p=0.95,
                                     max_tokens=100)
    llm = LLM(QWEN_7B_MODEL, max_num_seqs=18, max_num_batched_tokens=30)
    outputs = llm.generate(prompts, sampling_params)
    logger.info(outputs)
    for output in outputs:
        prompt = output.prompt
        assert prompt in prompts
        assert output.finished is True


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_ms_offline_chunked_prefill_004():
    """
    Test Summary:
        For the offline native backend scenario with default mode,
        Test sending combined requests for multiple scenarios.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    import vllm_mindspore
    from vllm import LLM
    llm = LLM(QWEN_7B_MODEL,
              max_num_seqs=16,
              max_num_batched_tokens=20,
              enable_chunked_prefill=True)
    test_results = run_combination_accuracy(llm=llm,
                                            batches=[1, 4],
                                            concurrency_levels=[1, 5],
                                            seq_lengths=[5, 50],
                                            formats=["prompt", "chat"],
                                            languages=["english", "chinese"],
                                            ignored_basic_check=False,
                                            model_max_token=32768)
    assert test_results.get('failure') == 0


def run_ms_server_cp_base_qwen(log_name, extra_params, prompts):
    model = QWEN_7B_MODEL
    process = start_vllm_server(model,
                                log_name,
                                start_mode='serve',
                                extra_params=extra_params)
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        url = f'http://localhost:{serve_port}/v1/completions'
    else:
        url = 'http://localhost:8000/v1/completions'

    data = {
        "model": model,
        "prompt": prompts,
        "max_tokens": 100,
        "top_k": -1,
        "top_p": 0.95,
        "temperature": 0.5
    }
    json_data = json.dumps(data)
    response = requests.post(url,
                             data=json_data,
                             headers={'Content-Type': 'application/json'})
    stop_vllm_server(process)
    assert response.status_code == 200
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1


@pytest.mark.level0
@patch.dict(os.environ, env_vars)
def test_vllm_ms_server_chunked_prefill_001():
    """
    Test Summary:
        For the online native backend scenario with default mode, TP2.
        Test sending single batch request whose length do not exceeding
        the max_num_batched_tokens.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    import vllm_mindspore
    prompts = "I love Beijing"
    log_name = "test_vllm_ms_server_chunked_prefill_001.log"
    extra_params = '--tensor_parallel_size=2 --enable-chunked-prefill '
    '--max_num_seqs 16 --max-num-batched-tokens 32 '

    run_ms_server_cp_base_qwen(log_name, extra_params, prompts)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_ms_server_chunked_prefill_002():
    """
    Test Summary:
        For the online native backend scenario with default mode,
        Test sending multiple batch requests, where all requests do not
        exceed max_num_batched_tokens.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    import vllm_mindspore
    prompts = ["I love Beijing", "Today is", "Llama is"]
    log_name = "test_vllm_ms_server_chunked_prefill_002.log"
    extra_params = '--max_num_seqs 16 --max-num-batched-tokens 32 '

    run_ms_server_cp_base_qwen(log_name, extra_params, prompts)


@pytest.mark.level1
@patch.dict(os.environ, env_vars)
def test_vllm_ms_server_chunked_prefill_003():
    """
    Test Summary:
        For the online native backend scenario with default mode,
        Test sending single batch request whose length exceeding the
        max_num_batched_tokens.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    import vllm_mindspore
    prompts = LONG_PROMPT
    log_name = "test_vllm_ms_server_chunked_prefill_003.log"
    extra_params = '--tensor_parallel_size=2 --max_num_seqs 16 '
    '--max-num-batched-tokens 32'

    run_ms_server_cp_base_qwen(log_name, extra_params, prompts)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_ms_server_chunked_prefill_004():
    """
    Test Summary:
        For the online native backend scenario with default mode,
        Test sending multiple batch requests, including the case of
        the request length exceeding the max_num_batched_tokens.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    import vllm_mindspore
    prompt1 = LONG_PROMPT
    prompt2 = "Today is"
    prompt3 = "Llama is"
    prompts = [prompt1, prompt2, prompt3]
    log_name = "test_vllm_ms_server_chunked_prefill_004.log"
    extra_params = '--max_num_seqs 16 --max-num-batched-tokens 32 '

    run_ms_server_cp_base_qwen(log_name, extra_params, prompts)


def run_server_chunked_prefill_005(log_name):
    process = start_vllm_server(
        QWEN_7B_MODEL,
        log_name,
        start_mode='serve',
        extra_params='--tensor_parallel_size=2 --max_num_seqs 16 '
        '--max-num-batched-tokens 32')
    test_results = run_combination_accuracy(model=QWEN_7B_MODEL,
                                            is_service=True,
                                            batches=[1, 4],
                                            concurrency_levels=[1, 5],
                                            seq_lengths=[5, 50],
                                            formats=["prompt", "chat"],
                                            languages=["english", "chinese"],
                                            ignored_basic_check=False,
                                            model_max_token=32768)
    stop_vllm_server(process)
    return test_results


@pytest.mark.level1
@patch.dict(os.environ, env_vars)
def test_vllm_ms_server_chunked_prefill_005():
    """
    Test Summary:
        For the online native backend scenario with default mode, TP2.
        Test sending combined requests for multiple scenarios.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    import vllm_mindspore
    log_name = "test_vllm_ms_server_chunked_prefill_005.log"
    test_results = run_server_chunked_prefill_005(log_name)
    assert test_results.get('failure') == 0
