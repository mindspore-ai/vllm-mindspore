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
"""
This example shows how to use the multi-LoRA functionality
for offline inference.

"""
import os
import json
import time
import requests
import pytest
from unittest.mock import patch
import vllm_mindspore
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from tests.utils.common_utils import (teardown_function, setup_function,
                                      MODEL_PATH, check_hit,
                                      get_key_counter_from_log,
                                      start_vllm_server, stop_vllm_server,
                                      run_combination_accuracy,
                                      process_request)

from typing import Optional

from vllm import EngineArgs, LLMEngine, RequestOutput

# def env
env_vars = {
    "VLLM_MS_MODEL_BACKEND": "Native",
    "MS_ENABLE_LCCL": "off",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
}

QWEN_7B_MODEL = MODEL_PATH["Qwen2.5-7B-Instruct"]
QWEN_7B_LORA_LAW = MODEL_PATH["Qwen2.5-7B-Lora-Law"]
QWEN_7B_LORA_MEDICAL = MODEL_PATH["Qwen2.5-7B-Lora-Medical"]


def create_test_prompts(
        lora_path: str
) -> list[tuple[str, SamplingParams, Optional[LoRARequest]]]:
    """Create a list of test prompts with their sampling parameters.
    """
    return [
        ("违章停车与违法停车是否有区别？",
         SamplingParams(temperature=0.0, top_p=1, top_k=-1,
                        max_tokens=10), LoRARequest("sql-lora1", 1,
                                                    lora_path)),
    ]


def process_requests(engine: LLMEngine,
                     test_prompts: list[tuple[str, SamplingParams,
                                              Optional[LoRARequest]]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, lora_request = test_prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               lora_request=lora_request)
            request_id += 1

        request_outputs: list[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                print(f'text is: {request_output.outputs[0].text}', flush=True)
                assert " 从法律上来说，违章停车和违法" in \
                    request_output.outputs[0].text


def initialize_engine() -> LLMEngine:
    """Initialize the LLMEngine."""
    # max_loras: controls the number of LoRAs that can be used in the same
    #   batch. Larger numbers will cause higher memory usage, as each LoRA
    #   slot requires its own preallocated tensor.
    # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
    #   numbers will cause higher memory usage. If you know that all LoRAs will
    #   use the same rank, it is recommended to set this as low as possible.
    # max_cpu_loras: controls the size of the CPU LoRA cache.
    engine_args = EngineArgs(model=QWEN_7B_MODEL,
                             enable_lora=True,
                             max_loras=1,
                             max_lora_rank=64,
                             max_cpu_loras=2,
                             max_num_seqs=256,
                             max_model_len=256,
                             max_num_batched_tokens=400)
    return LLMEngine.from_engine_args(engine_args)


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
def test_multilora_inference():
    """
    Test Summary:
        test function that sets up and runs the prompt processing.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Lora-Law
    """
    engine = initialize_engine()
    lora_path = QWEN_7B_LORA_LAW
    test_prompts = create_test_prompts(lora_path)
    process_requests(engine, test_prompts)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_ms_offline_multilora_002():
    """
    Test Summary:
        For the offline native backend scenario with default mode,
        do not set --enable-lora, and pass a LoRARequest.
    Expected Result:
        Raises ValueError with a reasonable information.
    Model Info:
        Qwen2.5-7B-Instruct, Qwen2.5-7B-Lora-Law
    """
    model = QWEN_7B_MODEL
    prompts = ["Hello,", "你好,"]
    sampling_params = SamplingParams(temperature=0.0,
                                     top_p=0.95,
                                     top_k=3,
                                     repetition_penalty=2.0)
    llm = LLM(model=model, tensor_parallel_size=1)
    with pytest.raises(ValueError) as err:
        llm.generate(prompts,
                     sampling_params,
                     lora_request=LoRARequest("lora1", 1, QWEN_7B_LORA_LAW))
    assert "LoRA is not enabled!" in str(err.value)


@pytest.mark.level0
@patch.dict(os.environ, env_vars)
def test_vllm_ms_offline_multilora_003():
    """
    Test Summary:
        For the offline native backend scenario with default mode,
        enable --enable-lora, and pass duplicate LoRA IDs in LoRARequest
        with different LoRAs.
    Expected Result:
        Normal execution without errors
    Model Info:
        Qwen2.5-7B-Instruct, Qwen2.5-7B-Lora-Law, Qwen2.5-7B-Lora-Medical
    """
    model = QWEN_7B_MODEL
    prompts = ["Hello,", "你好,"]
    sampling_params = SamplingParams(temperature=0.0,
                                     top_p=0.95,
                                     top_k=3,
                                     repetition_penalty=2.0)
    llm = LLM(model=model,
              max_lora_rank=64,
              max_loras=1,
              enable_lora=True,
              tensor_parallel_size=2)
    outputs = llm.generate(prompts,
                           sampling_params,
                           lora_request=[
                               LoRARequest("lora1", 1, QWEN_7B_LORA_LAW),
                               LoRARequest("lora2", 1, QWEN_7B_LORA_MEDICAL)
                           ])
    for i, output in enumerate(outputs):
        prompt = output.prompt
        assert prompt == prompts[i]
        assert output.finished is True


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_ms_offline_multilora_004():
    """
    Test Summary:
        For the offline native backend scenario with default mode,
        enable --enable-lora, and pass different LoRA IDs for the same
        LoRA in LoRARequest.
    Expected Result:
        Normal execution without errors
    Model Info:
        Qwen2.5-7B-Instruct, Qwen2.5-7B-Lora-Law, Qwen2.5-7B-Lora-Medical
    """
    model = QWEN_7B_MODEL
    prompts = ["Hello,", "你好,"]
    sampling_params = SamplingParams(temperature=0.0,
                                     top_p=0.95,
                                     top_k=3,
                                     repetition_penalty=2.0)
    llm = LLM(model=model,
              max_lora_rank=64,
              max_loras=1,
              enable_lora=True,
              max_model_len=1024,
              max_num_batched_tokens=1024)
    outputs = llm.generate(prompts,
                           sampling_params,
                           lora_request=[
                               LoRARequest("lora1", 1, QWEN_7B_LORA_LAW),
                               LoRARequest("lora1", 2, QWEN_7B_LORA_MEDICAL)
                           ])
    for i, output in enumerate(outputs):
        prompt = output.prompt
        assert prompt == prompts[i]
        assert output.finished is True


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_ms_offline_multilora_005():
    """
    Test Summary:
        For the offline native backend scenario with default mode,
        enable --enable-lora, and pass an invalid lora_path in
        LoRARequest.
    Expected Result:
        Raised Exception with a reasonable information.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_ms_offline_multilora_005.log"
    parent_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_path = os.path.join(parent_dir, "utils", log_name)

    model = QWEN_7B_MODEL
    prompts = ["Hello,", "你好,"]
    lora_path = "/path/to/error/lora"
    sampling_params = SamplingParams(temperature=0.0,
                                     top_p=0.95,
                                     top_k=3,
                                     repetition_penalty=2.0)

    with open(log_path, "w") as f:
        original_stdout = os.dup(1)
        original_stderr = os.dup(2)
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)

        try:
            llm = LLM(model=model,
                      max_lora_rank=64,
                      max_loras=1,
                      enable_lora=True,
                      max_model_len=1024,
                      max_num_batched_tokens=1024,
                      tensor_parallel_size=1)

            llm.generate(prompts,
                         sampling_params,
                         lora_request=LoRARequest("lora1", 1, lora_path))
        except Exception as e:
            f.write(str(e) + "\n")
        finally:
            os.dup2(original_stdout, 1)
            os.dup2(original_stderr, 2)
            os.close(original_stdout)
            os.close(original_stderr)
    result = get_key_counter_from_log(
        log_name, "No adapter found for /path/to/error/lora")
    assert result >= 1


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_ms_server_multilora_001():
    """
    Test Summary:
        For the online native backend scenario with default mode,
        enable --enable-lora --lora-modules
    Expected Result:
        Execution successful, executed on the native backend
    Model Info:
        Qwen2.5-7B-Instruct, Qwen2.5-7B-Lora-Law
    """
    model = QWEN_7B_MODEL
    log_name = "test_vllm_ms_server_multilora_001.log"
    process = start_vllm_server(
        model,
        log_name,
        start_mode='serve',
        extra_params=f'--enable-lora --max_lora_rank=64 --max_loras=1 '
        f'--max_model_len=1024 --max_num_batched_tokens=1024 '
        f'--lora-modules lora1="{QWEN_7B_LORA_LAW}"')
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        url = f'http://localhost:{serve_port}/v1/completions'
    else:
        url = 'http://localhost:8000/v1/completions'
    data = {"model": model, "prompt": ["你好"], "top_k": 1, "top_p": 0.95}
    json_data = json.dumps(data)
    response = requests.post(url,
                             data=json_data,
                             headers={'Content-Type': 'application/json'})
    stop_vllm_server(process)
    assert response.status_code == 200
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_ms_server_multilora_002():
    """
    Test Summary:
        For the online native backend scenario with default mode,
        do not set --enable-lora, but set --lora-modules {name}={path}.
    Expected Result:
        Raises ValueError with a reasonable information.
    Model Info:
        Qwen2.5-7B-Instruct, Qwen2.5-7B-Lora-Law
    """
    model = QWEN_7B_MODEL
    log_name = "test_vllm_ms_server_multilora_002.log"
    process = start_vllm_server(
        model,
        log_name,
        start_mode='serve',
        normal_case=False,
        extra_params='--max_lora_rank=64 --max_loras=1 --max_model_len=1024 '
        '--max_num_batched_tokens=1024 '
        '--lora-modules lora1="{QWEN_7B_LORA_LAW}"')
    stop_vllm_server(process)
    result = get_key_counter_from_log(
        log_name, "ValueError: Call to add_lora method "
        "failed: LoRA is not enabled")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_ms_server_multilora_003():
    """
    Test Summary:
        For the online native backend scenario with default mode,
        enable --enable-lora, --lora-modules {name}={path},
        but the path does not exist.
    Expected Result:
        Raises ValueError with a reasonable information.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    model = QWEN_7B_MODEL
    log_name = "test_vllm_ms_server_multilora_003.log"
    process = start_vllm_server(
        model,
        log_name,
        start_mode='serve',
        normal_case=False,
        extra_params='--max_lora_rank=64 --max_loras=1 --max_model_len=1024 '
        '--enable-lora --max_num_batched_tokens=1024 '
        '--lora-modules lora1="/path/to/error/lora"')
    stop_vllm_server(process)
    result = get_key_counter_from_log(
        log_name, "ValueError: Call to add_lora method "
        "failed: Loading lora lora1 failed")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_ms_server_multilora_004():
    """
    Test Summary:
        For the online native backend scenario with default mode,
        enable --enable-lora, and repeatedly set --lora-modules {name}={path}
        configurations with the same name.
    Expected Result:
        Raises ValueError with a reasonable information.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    model = QWEN_7B_MODEL
    path = QWEN_7B_LORA_LAW
    path2 = QWEN_7B_LORA_MEDICAL
    log_name = "test_vllm_ms_server_multilora_004.log"
    process = start_vllm_server(
        model,
        log_name,
        start_mode='serve',
        normal_case=False,
        extra_params=f'--max_lora_rank=64 --max_loras=1 --max_model_len=1024 '
        f'--enable-lora --max_num_batched_tokens=1024 '
        f'--lora-modules lora1="{path}" lora1="{path2}" ')
    stop_vllm_server(process)
    result = get_key_counter_from_log(
        log_name,
        "ValueError: The lora adapter 'lora1' has already been loaded")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_ms_server_multilora_005():
    """
    Test Summary:
        For the online native backend scenario with default mode,
        enable --enable-lora, and repeatedly set --lora-modules {name}={path}
        configurations with the same path.
    Expected Result:
        Execution successful, executed on the native backend
    Model Info:
        Qwen2.5-7B-Instruct, Qwen2.5-7B-Lora-Law
    """
    model = QWEN_7B_MODEL
    path = QWEN_7B_LORA_LAW
    log_name = "test_vllm_ms_server_multilora_005.log"
    process = start_vllm_server(
        model,
        log_name,
        start_mode='serve',
        extra_params=f'--max_lora_rank=64 --max_loras=2 --max_model_len=1024 '
        f'--enable-lora --max_num_batched_tokens=1024 '
        f'--lora-modules lora1="{path}" lora2="{path}" ')
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        url = f'http://localhost:{serve_port}/'
    else:
        url = 'http://localhost:8000/'
    response = process_request("lora1", url, 2, 2, "prompt")
    response = process_request("lora2", url, 2, 2, "prompt")
    stop_vllm_server(process)
    for r in response:
        assert r not in ['404', '400', '500']
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1
    assert get_key_counter_from_log(log_name, path)


@pytest.mark.level1
@patch.dict(os.environ, env_vars)
def test_vllm_ms_server_multilora_007():
    """
    Test Summary:
        For the online native backend scenario with default mode,
        testing the multi-LoRA, APC, and CP features.
    Expected Result:
        Execution successful, the prefix cache is expected to hit.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    model = QWEN_7B_MODEL
    log_name = "test_vllm_ms_server_multilora_007.log"
    process = start_vllm_server(
        model,
        log_name,
        start_mode='serve',
        extra_params=f'--tensor_parallel_size=2 --enable-lora '
        f'--max_lora_rank=64 --max_model_len=4096 '
        f'--lora-modules lora1={QWEN_7B_LORA_LAW} '
        f'--enable-prefix-caching --block_size=16 '
        f'--enable-chunked-prefill --max_num_seqs 16 '
        f'--max-num-batched-tokens 32')
    test_results = run_combination_accuracy(model="lora1",
                                            is_service=True,
                                            batches=[3],
                                            seq_lengths=[100],
                                            formats=["prompt"],
                                            languages=["chinese"],
                                            concurrency_levels=[3],
                                            model_max_token=3276800,
                                            skip_mixed=True)
    time.sleep(10)
    stop_vllm_server(process)
    assert test_results.get("failure") == 0
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1
    assert check_hit(log_name)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_ms_offline_multilora_v1_001():
    """
    Test Summary:
        For the offline native backend scenario with default mode,
        enable enable_lora, and pass a valid LoRARequest.
    Expected Result:
        Execution successful
    Model Info:
        Qwen2.5-7B-Instruct, Qwen2.5-7B-Lora-Law, Qwen2.5-7B-Lora-Medical
    """
    model = QWEN_7B_MODEL
    prompts = ["Hello,", "你好,"]
    sampling_params = SamplingParams(temperature=0.0,
                                     top_p=0.95,
                                     top_k=3,
                                     repetition_penalty=2.0)
    llm = LLM(model=model,
              max_lora_rank=64,
              max_loras=2,
              enable_lora=True,
              max_model_len=1024,
              max_num_batched_tokens=1024)
    outputs = llm.generate(prompts,
                           sampling_params,
                           lora_request=[
                               LoRARequest("lora1", 1, QWEN_7B_LORA_LAW),
                               LoRARequest("lora2", 2, QWEN_7B_LORA_MEDICAL)
                           ])
    for i, output in enumerate(outputs):
        prompt = output.prompt
        assert prompt == prompts[i]
        assert output.finished is True


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_ms_server_multilora_v1_001():
    """
    Test Summary:
        For the online native backend scenario with default mode,
        enable --enable-lora, set --lora-modules lora1=xxx lora2=xxx,
        and then send mixed requests for the base model, lora1, and lora2.
    Expected Result:
        Execution successful
    Model Info:
        Qwen2.5-7B-Instruct, Qwen2.5-7B-Lora-Law, Qwen2.5-7B-Lora-Medical
    """
    model = QWEN_7B_MODEL
    log_name = "test_vllm_ms_server_multilora_006.log"
    process = start_vllm_server(
        model,
        log_name,
        start_mode='serve',
        extra_params=f'--enable-log-requests --enable-lora --max_lora_rank=64 '
        f'--max_model_len=4096 --lora-modules '
        f'lora1={QWEN_7B_LORA_LAW} lora2={QWEN_7B_LORA_MEDICAL}')
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        url = f'http://localhost:{serve_port}/'
    else:
        url = 'http://localhost:8000/'
    response = []
    response.append(process_request(model, url, 2, 2, "prompt"))
    response.append(process_request("lora1", url, 2, 2, "prompt"))
    response.append(process_request("lora2", url, 2, 2, "prompt"))
    stop_vllm_server(process)
    for r in response:
        assert r not in ['404', '400', '500']
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1
    assert get_key_counter_from_log(log_name, "lora1") > 5
    assert get_key_counter_from_log(log_name, "lora2") > 5
