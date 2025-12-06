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
import time
import json
import requests
from unittest.mock import patch
import vllm_mindspore

from tests.utils.common_utils import (teardown_function, setup_function,
                                      MODEL_PATH, check_hit,
                                      get_key_counter_from_log,
                                      start_vllm_server, stop_vllm_server,
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


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_ms_server_apc_002():
    """
    Test Summary:
        For the online native backend scenario with default mode,
        and then send multiple batches of requests.
    Expected Result:
        Successful execution, and the prefix cache is successfully hit.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_ms_server_apc_002.log"
    model = QWEN_7B_MODEL
    process = start_vllm_server(model,
                                log_name,
                                extra_params='--block_size=16')
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        url = f'http://localhost:{serve_port}/v1/completions'
    else:
        url = 'http://localhost:8000/v1/completions'
    data = {
        "model": model,
        "prompt": ["这是一篇新闻: 买商品、购服务，随着生产生活秩序恢复，"
                   "消费市场逐渐升温。物价水平怎么样？"] * 100,
        "top_k": 1
    }
    json_data = json.dumps(data)
    repeat_time = 2
    for _ in range(repeat_time):
        response = requests.post(url,
                                 data=json_data,
                                 headers={'Content-Type': 'application/json'})
    time.sleep(30)
    stop_vllm_server(process)
    assert response.status_code == 200
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1
    assert check_hit(log_name)


@pytest.mark.level1
@patch.dict(os.environ, env_vars)
def test_vllm_ms_server_apc_003():
    """
    Test Summary:
        For the online native backend scenario with default mode,
        and perform precision verification using utility tools.
    Expected Result:
        Successful execution, and the prefix cache is successfully hit.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_ms_server_apc_003.log"
    model = QWEN_7B_MODEL
    process = start_vllm_server(
        model,
        log_name,
        start_mode='serve',
        extra_params='--tensor_parallel_size=2 --block_size=16')
    test_results = run_combination_accuracy(model=model,
                                            is_service=True,
                                            batches=[3],
                                            seq_lengths=[1000],
                                            concurrency_levels=[3],
                                            model_max_token=3276800)
    time.sleep(10)
    stop_vllm_server(process)
    assert test_results.get("failure") < 3
    assert check_hit(log_name)
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_ms_server_apc_004():
    """
    Test Summary:
        For the online native backend scenario, disable the APC feature,
        and then send two consecutive requests.
    Expected Result:
        Successful execution, but the prefix cache has no hits.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_ms_server_apc_004.log"
    model = QWEN_7B_MODEL
    process = start_vllm_server(model,
                                log_name,
                                start_mode='serve',
                                extra_params='--no-enable-prefix-caching '
                                '--block_size=16')
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        url = f'http://localhost:{serve_port}/v1/completions'
    else:
        url = 'http://localhost:8000/v1/completions'
    data = {
        "model": model,
        "prompt": ["这是一篇新闻: 买商品、购服务，随着生产生活秩序恢复，"
                   "消费市场逐渐升温。物价水平怎么样？"],
        "top_k": 1
    }
    json_data = json.dumps(data)
    repeat_time = 2
    for _ in range(repeat_time):
        response = requests.post(url,
                                 data=json_data,
                                 headers={'Content-Type': 'application/json'})
    time.sleep(30)
    assert response.status_code == 200
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1
    assert not check_hit(log_name)
    stop_vllm_server(process)
