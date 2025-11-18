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

from tests.st.python.utils.env_var_manager import EnvVarManager

env_manager = EnvVarManager()
env_manager.setup_mindformers_environment()
env_vars = {
    "vLLM_MODEL_BACKEND": "MindFormers",
    "MS_ENABLE_LCCL": "off",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    "MS_ALLOC_CONF": "enable_vmm:True",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
    "HCCL_IF_BASE_PORT": "60000",
    "LCAL_COMM_ID": "127.0.0.1:10068"
}

DS_R1_W8A8_MODEL = "/home/workspace/mindspore_dataset/weight/DeepSeek-R1-W8A8"


def execute_shell_command(command):
    """执行 shell 命令并返回状态和输出"""
    from vllm.logger import init_logger
    logger = init_logger(__name__)
    status, output = subprocess.getstatusoutput(command)
    if status != 0:
        logger.info("执行命令失败: %s\n错误信息: %s", command, output)
    return status, output


def stop_vllm_server(process=None):
    """停止 vLLM 服务及其相关进程"""
    from vllm.logger import init_logger
    logger = init_logger(__name__)
    if process is not None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
        except Exception as e:
            logger.info("终止进程组失败: %s", e)

    commands = [
        "npu-smi info | grep python3 | awk '{print $5}'",
        "npu-smi info | grep vllm-mindspore | awk '{print $5}'",
        "ps -ef | grep vllm-mindspore | grep -v grep | awk '{print $2}'",
        "ps -ef | grep scheduler_init.py | grep -v grep | awk '{print $2}'",
        "ps -ef | grep -E '(python3|python)' | grep entrypoint | grep -v grep "
        "| awk '{print $2}'",
        "ps -ef | grep -E '(python3|python)' | grep 'from multiprocessing.' "
        "| grep -v grep | awk '{print $2}'"
    ]

    for cmd in commands:
        status, output = execute_shell_command(cmd)
        if status == 0 and output.strip():
            kill_cmd = f"kill -9 {output.strip()}"
            execute_shell_command(kill_cmd)

    execute_shell_command("ray stop")
    time.sleep(10)


def get_key_counter_from_log(log_name, key):
    dirname, _ = os.path.split(os.path.abspath(__file__))
    log_path = os.path.join(dirname, log_name)
    if "'" in key:
        cmd = f"cat {log_path}|grep \"{key}\"|wc -l"
    else:
        cmd = f"cat {log_path}|grep '{key}'|wc -l"
    _, result = subprocess.getstatusoutput(cmd)
    return int(result)


def start_vllm_server(model, log_name, extra_params=''):
    """
    启动vllm服务函数
    Args:
        model: 请求中的model名称
        log_name: 服务拉起日志文件名称
        extra_params: 额外启动参数
    Returns:
        process: 拉起服务的进程号
    """
    from vllm.logger import init_logger
    logger = init_logger(__name__)
    dirname, _ = os.path.split(os.path.abspath(__file__))
    log_path = os.path.join(dirname, log_name)
    start_cmd = f"vllm-mindspore serve {model}"
    cmd = f"{start_cmd} " + \
          f"{extra_params} > {log_path} 2>&1"
    logger.info(cmd)
    process = subprocess.Popen(cmd,
                               shell=True,
                               executable='/bin/bash',
                               stdout=None,
                               stderr=None,
                               preexec_fn=os.setsid)

    time.sleep(10)
    count = 0
    cycle_time = 50
    while count < cycle_time:
        result = get_key_counter_from_log(log_name,
                                          "Application startup complete")
        if result > 0:
            break
        result = get_key_counter_from_log(log_name, "ERROR")
        if result > 0:
            stop_vllm_server()
            with open(log_path) as f:
                err_log = f.read()
            raise RuntimeError("vllm server fails to start!" + str(err_log))
            break
        time.sleep(10)
        count += 1
    else:
        stop_vllm_server()
        with open(log_path) as f:
            err_log = f.read()
        raise RuntimeError("vllm server fails to start!" + str(err_log))
    return process


def set_request(model_path, master_ip="127.0.0.1", port="8000"):
    from vllm.logger import init_logger
    logger = init_logger(__name__)
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

    process = start_vllm_server(model, log_name, extra_params=server_params)

    set_request(model, master_ip=dp_master_ip, port=server_port)
    stop_vllm_server(process)
    if os.path.exists(log_path):
        os.remove(log_path)
