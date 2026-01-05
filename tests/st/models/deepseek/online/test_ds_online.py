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

# isort:skip_file
"""test vllm deepseek online server."""
import pytest
from unittest.mock import patch
import os
import json
import requests
import subprocess
import shlex
import signal
import time

from tests.utils.common_utils import (teardown_function, setup_function,
                                      MODEL_PATH, start_vllm_server,
                                      get_key_counter_from_log,
                                      stop_vllm_server, logger)

env_vars = {
    "vLLM_MODEL_BACKEND": "MindFormers",
    "MS_ENABLE_LCCL": "off",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
    "HCCL_IF_BASE_PORT": "60000",
    "LCAL_COMM_ID": "127.0.0.1:10068"
}

DS_R1_W8A8_MODEL = MODEL_PATH["DeepSeek-R1-W8A8"]


def set_request(model_path, master_ip="127.0.0.1", port="8000"):
    url = f"http://{master_ip}:{port}/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model":
        model_path,
        "prompt":
        "You are a helpful assistant.<｜User｜>将文本分类为中性、"
        "负面或正面。 \n文本：我认为这次假期还可以。 \n情感："
        "<｜Assistant｜>\n",
        "max_tokens":
        3,  # 期望输出的token长度
        "temperature":
        0,
        "top_p":
        1.0,
        "top_k":
        1,
        "repetition_penalty":
        1.0
    }
    expect_result = 'ugs611ాలు'

    time_start = time.time()
    response = requests.post(url, headers=headers, json=data)
    res_time = round(time.time() - time_start, 2)
    try:
        generate_text = (json.loads(
            response.text).get("choices")[0].get("text"))
    except (json.JSONDecodeError, AttributeError):
        generate_text = ""

    logger.info("request: %s", data)
    logger.info("response: %s", response)
    logger.info("response.text: %s", response.text)
    logger.info("generate_text: %s", generate_text)
    logger.info("res_time: %s", res_time)
    assert generate_text == expect_result


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.allcards
def test_deepseek_r1_dp4_tp2_ep4_online():
    """
    Test Summary:
        test deepseek r1 with dp4 tp2 ep4
    Expected Result:
        Start online service successfully and send request, the first three
        tokens in the return result.
    Model Info:
        DeepSeek-R1-W8A8
    """
    import vllm_mindspore
    from vllm.utils import get_open_port  # noqa: E402

    log_name = "test_deepseek_r1_dp4_tp2_ep4_online.log"
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            log_name)

    model = DS_R1_W8A8_MODEL
    quant_type = 'ascend'
    dp_master_ip = "127.0.0.1"
    server_port = "8000"
    dp_master_port = shlex.quote(str(get_open_port()))
    stop_vllm_server()

    server_params = f"--trust_remote_code "\
                    f"--max-num-seqs=8 "\
                    f"--max_model_len=4096 "\
                    f"--max-num-batched-tokens=8 "\
                    f"--block-size=128 "\
                    f"--gpu-memory-utilization=0.7 "\
                    f"--quantization {quant_type} "\
                    f"--tensor-parallel-size 2 "\
                    f"--data-parallel-size 4 "\
                    f"--data-parallel-size-local 4 "\
                    f"--data-parallel-start-rank 0 "\
                    f"--data-parallel-address {dp_master_ip} "\
                    f"--data-parallel-rpc-port {dp_master_port} "\
                    f"--enable-expert-parallel "\
                    f"--additional-config '{{\"expert_parallel\": 4}}'"

    process = start_vllm_server(model,
                                log_name,
                                start_mode='serve',
                                extra_params=server_params)
    set_request(model, master_ip=dp_master_ip, port=server_port)

    stop_vllm_server(process)
    if os.path.exists(log_path):
        os.remove(log_path)
