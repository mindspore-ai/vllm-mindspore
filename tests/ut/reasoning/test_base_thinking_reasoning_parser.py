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
from unittest.mock import patch
from tests.utils.common_utils import (teardown_function, setup_function,
                                      MODEL_PATH, start_vllm_server,
                                      get_key_counter_from_log,
                                      stop_vllm_server, send_and_get_request)
from tests.utils.env_var_manager import EnvVarManager

env_manager = EnvVarManager()
env_manager.setup_mindformers_environment()

# def env
env_vars = {
    "VLLM_MS_MODEL_BACKEND": "MindFormers",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
}

DEEPSEEK_W8A8_MODEL = MODEL_PATH["DeepSeek-R1-W8A8"]


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_reasoning_mf_server():
    """
    Test Summary:
        For the online mindformers backend scenario, and set
        --reasoning-parser deepseek_r1.
    Expected Result:
        Successful execution, enabling the reasoning-parser allows the model
        to start reasoning.
    Model Info:
        DeepSeek-R1-W8A8
    """
    log_name = "test_vllm_reasoning_mf_server.log"
    model = DEEPSEEK_W8A8_MODEL
    process = start_vllm_server(model,
                                log_name,
                                extra_params="--reasoning-parser deepseek_r1 "
                                "--max-model-len=32768")
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
        100
    }
    response = send_and_get_request(data, fmt="chat")
    stop_vllm_server(process)
    assert response.status_code == 200, response.text
    assert len(
        response.json()["choices"][0]["message"]["reasoning_content"]) >= 100
    result = get_key_counter_from_log(log_name, "Run with Mindformers backend")
    assert result >= 1
