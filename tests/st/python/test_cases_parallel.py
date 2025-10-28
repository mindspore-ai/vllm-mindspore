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
"""test cases parallel"""

import os
from multiprocessing.pool import Pool

import pytest

from .utils import cleanup_subprocesses, tasks_resource_alloc


def teardown_function():
    cleanup_subprocesses()


def run_command(command_info):
    cmd, log_path = command_info
    ret = os.system(cmd)
    return ret, log_path


def check_results(commands, results):
    error_idx = [_ for _ in range(len(results)) if results[_][0] != 0]
    for idx in error_idx:
        print(f"testcase {commands[idx]} failed. "
              f"Please check log {results[idx][1]}.")
        os.system(f"grep -E 'ERROR|error|Error' {results[idx][1]} -C 5")
    assert error_idx == []


def run_tasks(cases):
    commands = tasks_resource_alloc(cases)

    with Pool(len(commands)) as pool:
        results = list(pool.imap(run_command, commands))
    check_results(commands, results)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_cases_parallel_part0():
    """
    Feature: test cases parallel.
    Description: test cases parallel.
    Expectation: Pass.
    """
    cases = [
        (2, "cases_parallel/vllm_mf_qwen_7b.py::test_mf_qwen",
         "vllm_mf_qwen_7b_test_mf_qwen.log"),
        (2, "cases_parallel/vllm_mf_qwen_7b_chunk_prefill.py"
         "::test_mf_qwen_7b_chunk_prefill",
         "vllm_mf_qwen_7b_chunk_prefill_test_mf_qwen_7b_chunk_prefill.log"),
        (2, "cases_parallel/vllm_mf_qwen_7b_chunk_prefill_v1.py"
         "::test_mf_qwen_7b_chunk_prefill",
         "vllm_mf_qwen_7b_chunk_prefill_v1_test_mf_qwen_7b_chunk_prefill.log"),
        (2, "cases_parallel/vllm_sampling.py::test_vllm_sampling_n_logprobs",
         "vllm_sampling_n_logprobs_v1.log")
    ]
    run_tasks(cases)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_cases_parallel_part1():
    """
    Feature: test cases parallel.
    Description: test cases parallel.
    Expectation: Pass.
    """
    cases = [
        (2, "cases_parallel/vllm_mf_qwen_7b_mss.py::test_mf_qwen_7b_mss",
         "vllm_mf_qwen_7b_mss_test_mf_qwen_7b_mss.log"),
        (2, "cases_parallel/vllm_mf_qwen_7b_prefix_caching.py"
         "::test_mf_qwen_7b_prefix_caching",
         "vllm_mf_qwen_7b_prefix_caching_test_mf_qwen_7b_prefix_caching.log"),
        (2, "cases_parallel/vllm_mf_qwen_7b_prefix_caching_v1.py"
         "::test_mf_qwen_7b_prefix_caching",
         "vllm_mf_qwen_7b_prefix_caching_v1_test_mf_qwen_7b_prefix_caching.log"
         ),
        (2, "cases_parallel/vllm_mf_qwen_7b_v1.py::test_mf_qwen",
         "vllm_mf_qwen_7b_v1_test_mf_qwen.log")
    ]
    run_tasks(cases)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_cases_parallel_part2():
    """
    Feature: test cases parallel.
    Description: test cases parallel.
    Expectation: Pass.
    """
    cases = [(2, "cases_parallel/vllm_qwen_7b.py::test_vllm_qwen",
              "vllm_qwen_7b_test_vllm_qwen.log"),
             (2, "cases_parallel/vllm_qwen_7b_v1.py::test_vllm_qwen",
              "vllm_qwen_7b_v1_test_vllm_qwen.log"),
             (4, "cases_parallel/shm_broadcast.py::test_shm_broadcast",
              "shm_broadcast_test_shm_broadcast.log")]
    run_tasks(cases)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_cases_parallel_part3():
    """
    Feature: test cases parallel.
    Description: test cases parallel.
    Expectation: Pass.
    """
    cases = [
        (2, "cases_parallel/vllm_deepseek_bf16_part.py::test_deepseek_r1_bf16",
         "vllm_deepseek_bf16_part_test_deepseek_r1_bf16.log"),
        (2,
         "cases_parallel/vllm_deepseek_bf16_part_v1.py::test_deepseek_r1_bf16",
         "vllm_deepseek_bf16_part_v1_test_deepseek_r1_bf16.log"),
        (2, "cases_parallel/vllm_deepseek_part.py::test_deepseek_r1",
         "vllm_deepseek_part_test_deepseek_r1.log"),
        (2, "cases_parallel/vllm_deepseek_part_v1.py::test_deepseek_r1",
         "vllm_deepseek_part_v1_test_deepseek_r1.log"),
    ]
    run_tasks(cases)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_cases_parallel_part4():
    """
    Feature: test cases parallel.
    Description: test cases parallel.
    Expectation: Pass.
    """
    cases = [
        (2, "cases_parallel/vllm_mf_qwen3_8b.py::test_mf_qwen3_v0",
         "vllm_mf_qwen3_8b_test_mf_qwen3.log"),
        (2, "cases_parallel/vllm_mf_qwen3_8b.py::test_mf_qwen3_v1",
         "vllm_mf_qwen3_8b_v1_test_mf_qwen3.log"),
        (1, "cases_parallel/vllm_mf_telechat2_7b.py::test_mf_telechat2_7b",
         "vllm_mf_telechat2_7b_test_mf_telechat2_7b.log"),
        (1, "cases_parallel/vllm_qwen3.py::test_vllm_qwen3_8b",
         "vllm_qwen3_test_vllm_qwen3_8b.log"),
        (1, "cases_parallel/vllm_qwen3.py::test_vllm_qwen3_0_6b",
         "vllm_qwen3_test_vllm_qwen3_0_6b.log"),
        (1, "cases_parallel/vllm_llama3.py::test_vllm_llama3_8b",
         "vllm_llama3_8b_test_vllm_llama3.log")
    ]
    run_tasks(cases)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_cases_parallel_part5():
    """
    Feature: test cases parallel.
    Description: test cases parallel.
    Expectation: Pass.
    """
    cases = [
        (2, "cases_parallel/multilora_inference.py::test_multilora_inference",
         "multilora_inference_test_multilora_inference.log"),
        (2, "cases_parallel/vllm_qwen_7b_v1.py::test_qwen_enforce_eager",
         "vllm_qwen_7b_v1_test_qwen_enforce_eager.log"),
        (2, "cases_parallel/vllm_deepseek_part.py::test_deepseek_mtp",
         "vllm_deepseek_part_test_deepseek_mtp.log"),
        (1, "cases_parallel/vllm_qwen3.py::test_qwen3_enforce_eager",
         "vllm_qwen3_test_qwen3_enforce_eager.log"),
    ]
    run_tasks(cases)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_cases_parallel_part6():
    """
    Feature: test cases parallel.
    Description: test cases parallel.
    Expectation: Pass.
    """
    cases = [
        (2, "cases_parallel/vllm_qwen3_moe.py::test_vllm_qwen3_30b_a3b",
         "test_vllm_qwen3_30b_a3b.log"),
        (2, "cases_parallel/vllm_qwen3_moe.py::test_vllm_qwen3_30b_a3b_eager",
         "test_vllm_qwen3_30b_a3b_eager.log"),
    ]
    run_tasks(cases)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_cases_parallel_part7():
    """
    Feature: test cases parallel.
    Description: test cases parallel.
    Expectation: Pass.
    """
    cases = [
        (2, "cases_parallel/vllm_qwen2_5_vl_7b_v1.py::test_qwen2_5_vl_7b_v1",
         "vllm_qwen2_5_vl_7b_v1.log"),
        (1, "cases_parallel/vllm_qwen2_5_vl_7b_v1.py"
         "::test_qwen2_5_vl_7b_v1_enforce_eager",
         "vllm_qwen2_5_vl_7b_v1_enforce_eager.log"),
        (1, "cases_parallel/vllm_qwen2_5_vl_7b_v1.py"
         "::test_qwen2_5_vl_7b_v1_video_infer",
         "vllm_qwen2_5_vl_7b_v1_video_infer.log"),
    ]
    run_tasks(cases)


@pytest.mark.level4
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_cases_parallel_level4_part0():
    """
    Feature: test cases parallel.
    Description:
        vllm_mf_qwen_7b_cp_pc_mss.py::test_mf_qwen_7b_cp_pc_mss:
            accuracy error happens occasionally
    Expectation: Pass.
    """
    cases = [(2, "cases_parallel/vllm_mf_qwen_7b_cp_pc_mss.py"
              "::test_mf_qwen_7b_cp_pc_mss",
              "vllm_mf_qwen_7b_cp_pc_mss_test_mf_qwen_7b_cp_pc_mss.log")]
    run_tasks(cases)


@pytest.mark.level4
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_cases_parallel_level4_mcore1():
    """
    Mcore currently does not support the following test cases,
    adjust the level to level 4 until it is re supported
    """
    cases = [
        (2, "cases_parallel/vllm_deepseek_osl.py::test_deepseek_r1",
         "vllm_deepseek_osl_test_deepseek_r1.log"),
        (2, "cases_parallel/vllm_deepseek_smoothquant.py::test_deepseek_r1",
         "vllm_deepseek_smoothquant_test_deepseek_r1.log"),
        (2, "cases_parallel/vllm_deepseek_smoothquant_mss.py"
         "::test_deepseek_r1_mss",
         "vllm_deepseek_smoothquant_mss_test_deepseek_r1_mss.log")
    ]
    run_tasks(cases)


@pytest.mark.level4
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_cases_parallel_level4_mcore2():
    """
    Mcore currently does not support the following test cases,
    adjust the level to level 4 until it is re supported
    """
    cases = [
        (2, "cases_parallel/vllm_deepseek_a8w4.py::test_deepseek_r1_a8w4",
         "vllm_deepseek_a8w4_test_deepseek_r1_a8w4.log"),
    ]
    run_tasks(cases)


@pytest.mark.level0
@pytest.mark.platform_ascend310p
@pytest.mark.env_single
def test_cases_parallel_310p_part0():
    """
    Feature: test cases parallel in 310p.
    Description: test cases parallel.
    Expectation: Pass.
    """
    cases = [
        (2, "cases_parallel/vllm_mf_qwen3_8b.py::test_mf_qwen3_v1_310p",
         "vllm_mf_qwen3_8b_v1_310p_test_mf_qwen3.log"),
    ]
    run_tasks(cases)
