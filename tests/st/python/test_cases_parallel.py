# type: ignore
# isort: skip_file
# !/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
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
# ============================================================================
"""test cases parallel"""
import os
from multiprocessing.pool import Pool

import pytest


def run_command(command_info):
    cmd, log_path = command_info
    ret = os.system(cmd)
    return ret, log_path


def check_results(commands, results):
    error_idx = [_ for _ in range(len(results)) if results[_][0] != 0]
    for idx in error_idx:
        print(
            f"testcase {commands[idx]} failed. Please check log {results[idx][1]}."
        )
        os.system(f"grep -E 'ERROR|error|Error' {results[idx][1]} -C 5")
    assert error_idx == []


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_cases_parallel_part0():
    """
    Feature: test cases parallel.
    Description: test cases parallel.
    Expectation: Pass.
    """
    commands = [
        ("export ASCEND_RT_VISIBLE_DEVICES=0,1 && export LCAL_COMM_ID=127.0.0.1:10068 && "
         "export HCCL_IF_BASE_PORT=61000 && "
         "pytest -s -v cases_parallel/vllm_mf_qwen_7b.py::test_mf_qwen > vllm_mf_qwen_7b_test_mf_qwen.log",
         "vllm_mf_qwen_7b_test_mf_qwen.log"),
        ("export ASCEND_RT_VISIBLE_DEVICES=2,3 && export LCAL_COMM_ID=127.0.0.1:10069 && "
         "export HCCL_IF_BASE_PORT=61002 && "
         "pytest -s -v cases_parallel/vllm_mf_qwen_7b_chunk_prefill.py::test_mf_qwen_7b_chunk_prefill "
         "> vllm_mf_qwen_7b_chunk_prefill_test_mf_qwen_7b_chunk_prefill.log",
         "vllm_mf_qwen_7b_chunk_prefill_test_mf_qwen_7b_chunk_prefill.log"),
        ("export ASCEND_RT_VISIBLE_DEVICES=4,5 && export LCAL_COMM_ID=127.0.0.1:10070 && "
         "export HCCL_IF_BASE_PORT=61004 &&"
         "pytest -s -v cases_parallel/vllm_mf_qwen_7b_chunk_prefill_v1.py::test_mf_qwen_7b_chunk_prefill "
         "> vllm_mf_qwen_7b_chunk_prefill_v1_test_mf_qwen_7b_chunk_prefill.log",
         "vllm_mf_qwen_7b_chunk_prefill_v1_test_mf_qwen_7b_chunk_prefill.log"),
        ("export ASCEND_RT_VISIBLE_DEVICES=6,7 && export LCAL_COMM_ID=127.0.0.1:10071 && "
         "export HCCL_IF_BASE_PORT=61006 && "
         "pytest -s -v cases_parallel/multilora_inference.py::test_multilora_inference "
         "> multilora_inference_test_multilora_inference.log",
         "multilora_inference_test_multilora_inference.log")
    ]

    with Pool(len(commands)) as pool:
        results = list(pool.imap(run_command, commands))
    check_results(commands, results)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_cases_parallel_part1():
    """
    Feature: test cases parallel.
    Description: test cases parallel.
    Expectation: Pass.
    """
    commands = [
        ("export ASCEND_RT_VISIBLE_DEVICES=0,1 && export LCAL_COMM_ID=127.0.0.1:10068 && "
         "export HCCL_IF_BASE_PORT=61000 && "
         "pytest -s -v cases_parallel/vllm_mf_qwen_7b_mss.py::test_mf_qwen_7b_mss "
         "> vllm_mf_qwen_7b_mss_test_mf_qwen_7b_mss.log",
         "vllm_mf_qwen_7b_mss_test_mf_qwen_7b_mss.log"),
        ("export ASCEND_RT_VISIBLE_DEVICES=2,3 && export LCAL_COMM_ID=127.0.0.1:10069 && "
         "export HCCL_IF_BASE_PORT=61002 && "
         "pytest -s -v cases_parallel/vllm_mf_qwen_7b_prefix_caching.py::test_mf_qwen_7b_prefix_caching "
         "> vllm_mf_qwen_7b_prefix_caching_test_mf_qwen_7b_prefix_caching.log",
         "vllm_mf_qwen_7b_prefix_caching_test_mf_qwen_7b_prefix_caching.log"),
        ("export ASCEND_RT_VISIBLE_DEVICES=4,5 && export LCAL_COMM_ID=127.0.0.1:10070 && "
         "export HCCL_IF_BASE_PORT=61004 && "
         "pytest -s -v cases_parallel/vllm_mf_qwen_7b_prefix_caching_v1.py::test_mf_qwen_7b_prefix_caching "
         "> vllm_mf_qwen_7b_prefix_caching_v1_test_mf_qwen_7b_prefix_caching.log",
         "vllm_mf_qwen_7b_prefix_caching_v1_test_mf_qwen_7b_prefix_caching.log"
         ),
        ("export ASCEND_RT_VISIBLE_DEVICES=6,7 && export LCAL_COMM_ID=127.0.0.1:10071 && "
         "export HCCL_IF_BASE_PORT=61006 && "
         "pytest -s -v cases_parallel/vllm_mf_qwen_7b_v1.py::test_mf_qwen > vllm_mf_qwen_7b_v1_test_mf_qwen.log",
         "vllm_mf_qwen_7b_v1_test_mf_qwen.log")
    ]

    with Pool(len(commands)) as pool:
        results = list(pool.imap(run_command, commands))
    check_results(commands, results)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_cases_parallel_part2():
    """
    Feature: test cases parallel.
    Description: test cases parallel.
    Expectation: Pass.
    """
    commands = [
        ("export ASCEND_RT_VISIBLE_DEVICES=0,1 && export LCAL_COMM_ID=127.0.0.1:10068 && "
         "export HCCL_IF_BASE_PORT=61000 && "
         "pytest -s -v cases_parallel/vllm_qwen_7b.py::test_vllm_qwen "
         "> vllm_qwen_7b_test_vllm_qwen.log",
         "vllm_qwen_7b_test_vllm_qwen.log"),
        ("export ASCEND_RT_VISIBLE_DEVICES=2,3 && export LCAL_COMM_ID=127.0.0.1:10069 && "
         "export HCCL_IF_BASE_PORT=61002 && "
         "pytest -s -v cases_parallel/vllm_qwen_7b_v1.py::test_vllm_qwen "
         "> vllm_qwen_7b_v1_test_vllm_qwen.log",
         "vllm_qwen_7b_v1_test_vllm_qwen.log"),
        ("export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 && export LCAL_COMM_ID=127.0.0.1:10070 && "
         "export HCCL_IF_BASE_PORT=61004 && "
         "pytest -s -v cases_parallel/shm_broadcast.py::test_shm_broadcast "
         "> shm_broadcast_test_shm_broadcast.log",
         "shm_broadcast_test_shm_broadcast.log")
    ]

    with Pool(len(commands)) as pool:
        results = list(pool.imap(run_command, commands))
    check_results(commands, results)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_cases_parallel_part3():
    """
    Feature: test cases parallel.
    Description: test cases parallel.
    Expectation: Pass.
    """
    commands = [
        ("export ASCEND_RT_VISIBLE_DEVICES=0,1 && export LCAL_COMM_ID=127.0.0.1:10068 && "
         "export HCCL_IF_BASE_PORT=61000 && "
         "pytest -s -v cases_parallel/vllm_deepseek_bf16_part.py::test_deepseek_r1_bf16 "
         "> vllm_deepseek_bf16_part_test_deepseek_r1_bf16.log",
         "vllm_deepseek_bf16_part_test_deepseek_r1_bf16.log"),
        ("export ASCEND_RT_VISIBLE_DEVICES=2,3 && export LCAL_COMM_ID=127.0.0.1:10069 && "
         "export HCCL_IF_BASE_PORT=61002 && "
         "pytest -s -v cases_parallel/vllm_deepseek_bf16_part_v1.py::test_deepseek_r1_bf16 "
         "> vllm_deepseek_bf16_part_v1_test_deepseek_r1_bf16.log",
         "vllm_deepseek_bf16_part_v1_test_deepseek_r1_bf16.log"),
        ("export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 && export LCAL_COMM_ID=127.0.0.1:10070 && "
         "export HCCL_IF_BASE_PORT=61004 && "
         "pytest -s -v cases_parallel/vllm_deepseek_gptq_a16w4.py::test_deepseek_r1_gptq_a16w4 "
         "> vllm_deepseek_gptq_a16w4_test_deepseek_r1_gptq_a16w4.log",
         "vllm_deepseek_gptq_a16w4_test_deepseek_r1_gptq_a16w4.log")
    ]

    with Pool(len(commands)) as pool:
        results = list(pool.imap(run_command, commands))
    check_results(commands, results)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_cases_parallel_part4():
    """
    Feature: test cases parallel.
    Description: test cases parallel.
    Expectation: Pass.
    """
    commands = [
        ("export ASCEND_RT_VISIBLE_DEVICES=0,1 && export LCAL_COMM_ID=127.0.0.1:10068 && "
         "export HCCL_IF_BASE_PORT=61000 && "
         "pytest -s -v cases_parallel/vllm_deepseek_osl.py::test_deepseek_r1_mss "
         "> vllm_deepseek_osl_test_deepseek_r1_mss.log",
         "vllm_deepseek_osl_test_deepseek_r1_mss.log"),
        ("export ASCEND_RT_VISIBLE_DEVICES=2,3 && export LCAL_COMM_ID=127.0.0.1:10069 && "
         "export HCCL_IF_BASE_PORT=61002 && "
         "pytest -s -v cases_parallel/vllm_deepseek_part.py::test_deepseek_r1 "
         "> vllm_deepseek_part_test_deepseek_r1.log",
         "vllm_deepseek_part_test_deepseek_r1.log"),
        ("export ASCEND_RT_VISIBLE_DEVICES=4,5 && export LCAL_COMM_ID=127.0.0.1:10070 && "
         "export HCCL_IF_BASE_PORT=61004 && "
         "pytest -s -v cases_parallel/vllm_deepseek_part.py::test_deepseek_mtp "
         "> vllm_deepseek_part_test_deepseek_mtp.log",
         "vllm_deepseek_part_test_deepseek_mtp.log"),
        ("export ASCEND_RT_VISIBLE_DEVICES=6,7 && export LCAL_COMM_ID=127.0.0.1:10071 && "
         "export HCCL_IF_BASE_PORT=61006 && "
         "pytest -s -v cases_parallel/vllm_deepseek_part_v1.py::test_deepseek_r1 "
         "> vllm_deepseek_part_v1_test_deepseek_r1.log",
         "vllm_deepseek_part_v1_test_deepseek_r1.log")
    ]

    with Pool(len(commands)) as pool:
        results = list(pool.imap(run_command, commands))
    check_results(commands, results)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_cases_parallel_part5():
    """
    Feature: test cases parallel.
    Description: test cases parallel.
    Expectation: Pass.
    """
    commands = [
        ("export ASCEND_RT_VISIBLE_DEVICES=0,1 && export LCAL_COMM_ID=127.0.0.1:10068 && "
         "export HCCL_IF_BASE_PORT=61000 && "
         "pytest -s -v cases_parallel/vllm_mf_qwen3_8b.py::test_mf_qwen3 "
         "> vllm_mf_qwen3_8b_test_mf_qwen3.log",
         "vllm_mf_qwen3_8b_test_mf_qwen3.log"),
        ("export ASCEND_RT_VISIBLE_DEVICES=2,3 && export LCAL_COMM_ID=127.0.0.1:10069 && "
         "export HCCL_IF_BASE_PORT=61002 && "
         "pytest -s -v cases_parallel/vllm_mf_qwen3_8b_v1.py::test_mf_qwen3 "
         "> vllm_mf_qwen3_8b_v1_test_mf_qwen3.log",
         "vllm_mf_qwen3_8b_v1_test_mf_qwen3.log"),
        ("export ASCEND_RT_VISIBLE_DEVICES=4 && export LCAL_COMM_ID=127.0.0.1:10070 && "
         "export HCCL_IF_BASE_PORT=61004 && "
         "pytest -s -v cases_parallel/vllm_llama3.py::test_vllm_llama3_8b "
         "> vllm_llama3_8b_test_vllm_llama3.log",
         "vllm_llama3_8b_test_vllm_llama3.log"),
        ("export ASCEND_RT_VISIBLE_DEVICES=5 && export LCAL_COMM_ID=127.0.0.1:10071 && "
         "export HCCL_IF_BASE_PORT=61006 && "
         "pytest -s -v cases_parallel/vllm_llama3.py::test_vllm_llama3_1b "
         "> vllm_llama3_1b_test_vllm_llama3.log",
         "vllm_llama3_1b_test_vllm_llama3.log"),
    ]

    with Pool(len(commands)) as pool:
        results = list(pool.imap(run_command, commands))
    check_results(commands, results)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_cases_parallel_level1_part0():
    """
    Feature: test cases parallel.
    Description: test cases parallel.
    Expectation: Pass.
    """
    commands = [
        ("export ASCEND_RT_VISIBLE_DEVICES=0,1 && export LCAL_COMM_ID=127.0.0.1:10068 && "
         "export HCCL_IF_BASE_PORT=61000 && "
         "pytest -s -v cases_parallel/vllm_mf_qwen_7b_cp_pc_mss.py::test_mf_qwen_7b_cp_pc_mss "
         "> vllm_mf_qwen_7b_cp_pc_mss_test_mf_qwen_7b_cp_pc_mss.log",
         "vllm_mf_qwen_7b_cp_pc_mss_test_mf_qwen_7b_cp_pc_mss.log"),
        ("export ASCEND_RT_VISIBLE_DEVICES=2,3 && export LCAL_COMM_ID=127.0.0.1:10069 && "
         "export HCCL_IF_BASE_PORT=61002 && "
         "pytest -s -v cases_parallel/vllm_deepseek_osl.py::test_deepseek_r1 "
         "> vllm_deepseek_osl_test_deepseek_r1.log",
         "vllm_deepseek_osl_test_deepseek_r1.log"),
        ("export ASCEND_RT_VISIBLE_DEVICES=4,5 && export LCAL_COMM_ID=127.0.0.1:10070 && "
         "export HCCL_IF_BASE_PORT=61004 && "
         "pytest -s -v cases_parallel/vllm_deepseek_smoothquant.py::test_deepseek_r1 "
         "> vllm_deepseek_smoothquant_test_deepseek_r1.log",
         "vllm_deepseek_smoothquant_test_deepseek_r1.log"),
        ("export ASCEND_RT_VISIBLE_DEVICES=6,7 && export LCAL_COMM_ID=127.0.0.1:10071 && "
         "export HCCL_IF_BASE_PORT=61006 && "
         "pytest -s -v cases_parallel/vllm_deepseek_smoothquant_mss.py::test_deepseek_r1_mss "
         "> vllm_deepseek_smoothquant_mss_test_deepseek_r1_mss.log",
         "vllm_deepseek_smoothquant_mss_test_deepseek_r1_mss.log")
    ]

    with Pool(len(commands)) as pool:
        results = list(pool.imap(run_command, commands))
    check_results(commands, results)
