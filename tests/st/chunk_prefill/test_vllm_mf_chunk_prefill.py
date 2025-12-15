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
import pytest
import json
import requests
from unittest.mock import patch
from tests.utils.common_utils import (teardown_function, setup_function,
                                      MODEL_PATH, get_key_counter_from_log,
                                      start_vllm_server, stop_vllm_server,
                                      run_combination_accuracy)
from tests.utils.env_var_manager import EnvVarManager
from tests.st.chunk_prefill.test_vllm_native_chunk_prefill import (
    LONG_PROMPT, run_server_chunked_prefill_005)

env_manager = EnvVarManager()
env_manager.setup_mindformers_environment()

QWEN_7B_MODEL = MODEL_PATH["Qwen2.5-7B-Instruct"]
DEEPSEEK_W8A8_MODEL = MODEL_PATH["DeepSeek-R1-W8A8"]

env_vars = {
    "VLLM_MS_MODEL_BACKEND": "MindFormers",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
}


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_mf_offline_chunked_prefill_002():
    """
    Test Summary:
        For the offline mindformers backend scenario with default mode,
        Test sending multiple batch requests, including the case of
        the request length exceeding the max_num_batched_tokens.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    import vllm_mindspore
    from vllm import LLM, SamplingParams
    prompt1 = LONG_PROMPT
    prompt2 = "Today is"
    prompt3 = "Llama is"
    prompts = [prompt1, prompt2, prompt3]
    sampling_params = SamplingParams(temperature=0.0,
                                     top_p=0.95,
                                     max_tokens=100)
    llm = LLM(QWEN_7B_MODEL, max_num_seqs=18, max_num_batched_tokens=30)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        assert prompt in prompts
        assert output.finished is True


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_mf_offline_chunked_prefill_003():
    """
    Test Summary:
        For the offline mindformers backend scenario with default mode,
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
                                     top_k=3,
                                     max_tokens=100)
    llm = LLM(QWEN_7B_MODEL, max_num_seqs=18, max_num_batched_tokens=30)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        assert prompt in prompts
        assert output.finished is True


@pytest.mark.level1
@patch.dict(os.environ, env_vars)
def test_vllm_mf_offline_chunked_prefill_004():
    """
    Test Summary:
        For the offline mindformers backend scenario with default mode,
        Test sending single batch request whose length exceeding the
        max_num_batched_tokens.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        DeepSeek-R1-W8A8
    """
    import vllm_mindspore
    from vllm import LLM, SamplingParams
    prompts = LONG_PROMPT
    sampling_params = SamplingParams(temperature=0.0,
                                     top_p=0.95,
                                     max_tokens=100)
    llm = LLM(DEEPSEEK_W8A8_MODEL,
              max_num_seqs=16,
              max_num_batched_tokens=32,
              enable_chunked_prefill=True,
              max_model_len=32768,
              tensor_parallel_size=2)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        assert prompt == prompts
        assert output.finished is True


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_mf_offline_chunked_prefill_006():
    """
    Test Summary:
        For the offline mindformers backend scenario with default mode,
        the model contains MLA layers. Test sending multiple batch
        requests, where all requests do not exceed max_num_batched_tokens.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        DeepSeek-R1-W8A8
    """
    import vllm_mindspore
    from vllm import LLM, SamplingParams
    prompts = ["I love Beijing", "Today is", "Llama is"]
    sampling_params = SamplingParams(temperature=0.0,
                                     top_p=0.95,
                                     max_tokens=100)
    llm = LLM(DEEPSEEK_W8A8_MODEL,
              max_num_seqs=18,
              max_num_batched_tokens=30,
              max_model_len=32768)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        assert prompt in prompts
        assert output.finished is True


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_mf_offline_chunked_prefill_007():
    """
    Test Summary:
        For the offline mindformers backend scenario with default mode,
        Test sending combined requests for multiple scenarios.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    import vllm_mindspore
    from vllm import LLM
    llm = LLM(QWEN_7B_MODEL, max_num_seqs=16, max_num_batched_tokens=20)
    test_results = run_combination_accuracy(llm=llm,
                                            is_service=False,
                                            batches=[1, 4],
                                            concurrency_levels=[1, 5],
                                            seq_lengths=[5, 50],
                                            formats=["prompt", "chat"],
                                            languages=["english", "chinese"],
                                            ignored_basic_check=False,
                                            model_max_token=32768)
    assert test_results.get('failure') == 0


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_mf_offline_chunked_prefill_008():
    """
    Test Summary:
        For the offline mindformers backend scenario with default mode,
        the model contains MLA layers. Test sending combined requests
        for multiple scenarios.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        DeepSeek-R1-W8A8
    """
    import vllm_mindspore
    from vllm import LLM
    llm = LLM(DEEPSEEK_W8A8_MODEL,
              max_num_seqs=16,
              max_num_batched_tokens=20,
              enable_chunked_prefill=True,
              max_model_len=32768)
    test_results = run_combination_accuracy(llm=llm,
                                            is_service=False,
                                            batches=[1, 4],
                                            concurrency_levels=[1, 5],
                                            seq_lengths=[5, 50],
                                            formats=["prompt", "chat"],
                                            languages=["english", "chinese"],
                                            ignored_basic_check=True,
                                            model_max_token=163840)
    assert test_results.get('failure') == 0


def run_mf_server_chunked_prefill(log_name, model, extra_params, data):
    """
    Perform chunked prefill validation on the mf backend of the
    specific model and data.

    Args:
      log_name: File name for redirecting service log
      model: Model name, same as in the request
      extra_params: Additional startup parameter
      data: Request data
    """
    import vllm_mindspore
    process = start_vllm_server(model, log_name, extra_params=extra_params)
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        url = f'http://localhost:{serve_port}/v1/completions'
    else:
        url = 'http://localhost:8000/v1/completions'

    json_data = json.dumps(data)
    response = requests.post(url,
                             data=json_data,
                             headers={'Content-Type': 'application/json'})
    stop_vllm_server(process)
    assert response.status_code == 200
    result = get_key_counter_from_log(log_name, "Run with Mindformers backend")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_mf_server_chunked_prefill_002():
    """
    Test Summary:
        For the online mindformers backend scenario with default mode,
        Test sending multiple batch requests, including the case of
        the request length exceeding the max_num_batched_tokens.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    import vllm_mindspore
    log_name = "test_vllm_mf_server_chunked_prefill_002.log"
    model = QWEN_7B_MODEL
    prompts = [LONG_PROMPT, "Today is", "Llama is"]
    data = {
        "model": model,
        "prompt": prompts,
        "max_tokens": 100,
        "temperature": 0
    }
    extra_params = '--max_num_seqs 16 --max_model_len=32768 '
    '--max-num-batched-tokens 32'

    run_mf_server_chunked_prefill(log_name, model, extra_params, data)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_mf_server_chunked_prefill_003():
    """
    Test Summary:
        For the online mindformers backend scenario with default mode,
        Test sending single batch request whose length exceeding the
        max_num_batched_tokens.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        DeepSeek-R1-W8A8
    """
    import vllm_mindspore
    log_name = "test_vllm_mf_server_chunked_prefill_003.log"
    model = DEEPSEEK_W8A8_MODEL
    prompts = LONG_PROMPT
    data = {
        "model": model,
        "prompt": prompts,
        "max_tokens": 100,
        "temperature": 0
    }
    extra_params = '--max_num_seqs 16 --max-num-batched-tokens 32 '
    '--max_model_len=32768'

    run_mf_server_chunked_prefill(log_name, model, extra_params, data)


@pytest.mark.level1
@patch.dict(os.environ, env_vars)
def test_vllm_mf_server_chunked_prefill_004():
    """
    Test Summary:
        For the online mindformers backend scenario with default mode,
        TP2. Test sending multiple batch requests, where all requests do
        not exceed max_num_batched_tokens.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        DeepSeek-R1-W8A8
    """
    import vllm_mindspore
    log_name = "test_vllm_mf_server_chunked_prefill_004.log"
    model = DEEPSEEK_W8A8_MODEL
    prompts = [LONG_PROMPT, "Today is", "Llama is"]
    data = {
        "model": model,
        "prompt": prompts,
        "max_tokens": 100,
        "top_p": 0.95,
        "temperature": 0.5
    }
    extra_params = '--tensor_parallel_size=2 --max_num_seqs 16 '
    '--max-num-batched-tokens 32 --max_model_len=32768'

    run_mf_server_chunked_prefill(log_name, model, extra_params, data)


@pytest.mark.level1
@patch.dict(os.environ, env_vars)
def test_vllm_mf_server_chunked_prefill_005():
    """
    Test Summary:
        For the online mindformers backend scenario with default mode,
        Test sending combined requests for multiple scenarios.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    import vllm_mindspore
    log_name = "test_vllm_mf_server_chunked_prefill_005.log"
    test_results = run_server_chunked_prefill_005(log_name)
    assert test_results.get('failure') <= 5
