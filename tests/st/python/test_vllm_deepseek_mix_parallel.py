#!/usr/bin/env python3
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
"""test mf deepseek r1."""
import pytest
import os
import tempfile
import re

from . import set_env
from multiprocessing import Process, Queue

env_manager = set_env.EnvVarManager()

env_vars = {
    "MINDFORMERS_MODEL_CONFIG": "./config/predict_deepseek_r1_671b_w8a8.yaml",
    "ASCEND_CUSTOM_PATH": os.path.expandvars("$ASCEND_HOME_PATH/../"),
    "vLLM_MODEL_BACKEND": "MindFormers",
    "MS_ENABLE_LCCL": "off",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    "MS_ALLOC_CONF": "enable_vmm:True",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
    "HCCL_IF_BASE_PORT": "61000",
    "LCAL_COMM_ID": "127.0.0.1:10068"
}
env_manager.setup_ai_environment(env_vars)
import vllm_mindspore
from vllm import LLM, SamplingParams
from vllm.utils import get_open_port


def dp_func(dp_size, local_dp_rank, global_dp_rank, dp_master_ip, dp_master_port,
            GPUs_per_dp_rank, prompts, except_list, result_q):
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    promts_per_rank = len(prompts) // dp_size
    start = global_dp_rank * promts_per_rank
    end = start + promts_per_rank
    prompts = prompts[start:end]
    except_list = except_list[start:end]
    if len(prompts) == 0:
        prompts = ["Placeholder"]
    print(f"DP rank {global_dp_rank} needs to process {len(prompts)} prompts")

    sampling_params = SamplingParams(temperature=0.0,
                                     top_p=1.0,
                                     top_k=1,
                                     repetition_penalty=1.0,
                                     max_tokens=3)

    # Create an LLM.
    llm = LLM(model="/home/workspace/mindspore_dataset/weight/DeepSeek-R1-W8A8",
              tensor_parallel_size=GPUs_per_dp_rank,
              max_model_len = 4096,
              max_num_batched_tokens=8,
              max_num_seqs=8,
              trust_remote_code=True,
              enforce_eager=True,
              enable_expert_parallel=True)
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"DP rank {global_dp_rank}, Prompt: {prompt!r}, "
              f"Generated text: {generated_text!r}")
        result_q.put(generated_text == except_list[i])


def exec_ds_with_dp(new_yaml, replaced_pattern, dp_size, tp_size, prompts, except_list):
    file = open('./config/predict_deepseek_r1_671b_w8a8.yaml', 'r')
    content = file.read()
    file.close()

    replace_data_parallel = re.compile(r'data_parallel: 1')
    replace_model_parallel = re.compile(r'model_parallel: 16')
    replace_expert_parallel = re.compile(r'expert_parallel: 1')

    content = replace_data_parallel.sub(replaced_pattern[0], content)
    content = replace_model_parallel.sub(replaced_pattern[1], content)
    content = replace_expert_parallel.sub(replaced_pattern[2], content)

    with tempfile.TemporaryDirectory() as tmp_dir:
        new_yaml_path = os.path.join(tmp_dir, new_yaml)
        with open(new_yaml_path, 'w') as f:
            f.write(content)
        env_manager.set_env_var("MINDFORMERS_MODEL_CONFIG", new_yaml_path)

        node_size = 1
        node_rank = 0
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()

        dp_per_node = dp_size // node_size

        result_q = Queue()
        procs = []
        for local_dp_rank, global_dp_rank in enumerate(
                range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)):
            proc = Process(target=dp_func,
                           args=(dp_size, local_dp_rank,
                                 global_dp_rank, dp_master_ip, dp_master_port,
                                 tp_size, prompts, except_list, result_q))
            proc.start()
            procs.append(proc)
        exit_code = 0

        for proc in procs:
            proc.join(timeout=180)
            if proc.exitcode is None:
                print(f"Killing process {proc.pid} that "
                      f"didn't stop within 3 minutes.")
                proc.kill()
                exit_code = 1
            elif proc.exitcode:
                exit_code = proc.exitcode

        assert exit_code == 0
        result = True
        for proc in procs:
            result = result and result_q.get()
        assert result

    # unset env
    env_manager.unset_all()


def exec_ds_without_dp(new_yaml, replaced_pattern, prompts, except_list):
    file = open('./config/predict_deepseek_r1_671b_w8a8.yaml', 'r')
    content = file.read()
    file.close()

    replace_data_parallel = re.compile(r'data_parallel: 1')
    replace_model_parallel = re.compile(r'model_parallel: 16')
    replace_expert_parallel = re.compile(r'expert_parallel: 1')

    content = replace_data_parallel.sub(replaced_pattern[0], content)
    content = replace_model_parallel.sub(replaced_pattern[1], content)
    content = replace_expert_parallel.sub(replaced_pattern[2], content)

    with tempfile.TemporaryDirectory() as tmp_dir:
        new_yaml_path = os.path.join(tmp_dir, new_yaml)
        with open(new_yaml_path, 'w') as f:
            f.write(content)
        env_manager.set_env_var("MINDFORMERS_MODEL_CONFIG", new_yaml_path)


        # Create a sampling params object.
        sampling_params = SamplingParams(temperature=0.0, max_tokens=3, top_k=1, top_p=1.0,
                                         repetition_penalty=1.0)

        # Create an LLM.
        llm = LLM(model="/home/workspace/mindspore_dataset/weight/DeepSeek-R1-W8A8",
                  trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=8, max_model_len=4096)
        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        outputs = llm.generate(prompts, sampling_params)
        # Print the outputs.
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            assert generated_text == except_list[i]

    # unset env
    env_manager.unset_all()



@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.allcards
def test_deepseek_r1_dp4_tp2_ep4():
    """
    test case deepseek r1 w8a8 dp4 tp2 ep4
    """

    new_yaml = "dp4_tp2_ep4.yaml"
    replaced_pattern = ['data_parallel: 4', 'model_parallel: 2', 'expert_parallel: 4']
    dp_size = 4
    tp_size = 2
    # Sample prompts.
    prompts = [
        "You are a helpful assistant.<｜User｜>将文本分类为中性、负面或正面。 \n文本：我认为这次假期还可以。 "
        "\n情感：<｜Assistant｜>\n",
    ] * 4

    except_list = ['ugs611ాలు'] * 4
    exec_ds_with_dp(new_yaml, replaced_pattern, dp_size, tp_size, prompts, except_list)


@pytest.mark.skip(reason="Currently does not support relevant communication fusion operators in 910b")
def test_deepseek_r1_dp8_tp1_ep8():
    """
    test case deepseek r1 w8a8 Dp8 tp1 ep8
    """

    new_yaml = "dp8_tp1_ep8.yaml"
    replaced_pattern = ['data_parallel: 8', 'model_parallel: 1', 'expert_parallel: 8']
    dp_size = 8
    tp_size = 1
    # Sample prompts.
    prompts = [
        "You are a helpful assistant.<｜User｜>将文本分类为中性、负面或正面。 \n文本：我认为这次假期还可以。 "
        "\n情感：<｜Assistant｜>\n",
    ] * 8

    except_list = ['ugs611ాలు'] * 8
    exec_ds_with_dp(new_yaml, replaced_pattern, dp_size, tp_size, prompts, except_list)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.allcards
def test_deepseek_r1_dp2_tp4_ep1():
    """
    test case deepseek r1 w8a8 dp2 tp4 ep1
    """

    new_yaml = "dp2_tp4_ep1.yaml"
    replaced_pattern = ['data_parallel: 2', 'model_parallel: 4', 'expert_parallel: 1']
    dp_size = 2
    tp_size = 4
    # Sample prompts.
    prompts = [
        "You are a helpful assistant.<｜User｜>将文本分类为中性、负面或正面。 \n文本：我认为这次假期还可以。 "
        "\n情感：<｜Assistant｜>\n",
    ] * 2

    except_list = ['ugs611ాలు'] * 2
    exec_ds_with_dp(new_yaml, replaced_pattern, dp_size, tp_size, prompts, except_list)


@pytest.mark.skip(reason="Currently does not support relevant communication fusion operators in 910b")
def test_deepseek_r1_dp4_tp2_ep8():
    """
    test case deepseek r1 w8a8 dp4 tp2 ep8
    """

    new_yaml = "dp4_tp2_ep8.yaml"
    replaced_pattern = ['data_parallel: 4', 'model_parallel: 2', 'expert_parallel: 8']
    dp_size = 4
    tp_size = 2
    # Sample prompts.
    prompts = [
        "You are a helpful assistant.<｜User｜>将文本分类为中性、负面或正面。 \n文本：我认为这次假期还可以。 "
        "\n情感：<｜Assistant｜>\n",
    ] * 4

    except_list = ['ugs611ాలు'] * 4
    exec_ds_with_dp(new_yaml, replaced_pattern, dp_size, tp_size, prompts, except_list)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.allcards
def test_deepseek_r1_dp8_tp1_ep1():
    """
    test case deepseek r1 w8a8 dp8 tp1 ep1
    """

    new_yaml = "dp8_tp1_ep1.yaml"
    replaced_pattern = ['data_parallel: 8', 'model_parallel: 1', 'expert_parallel: 1']
    dp_size = 8
    tp_size = 1
    # Sample prompts.
    prompts = [
        "You are a helpful assistant.<｜User｜>将文本分类为中性、负面或正面。 \n文本：我认为这次假期还可以。 "
        "\n情感：<｜Assistant｜>\n",
    ] * 8

    except_list = ['ugs611ాలు'] * 8
    exec_ds_with_dp(new_yaml, replaced_pattern, dp_size, tp_size, prompts, except_list)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.allcards
def test_deepseek_r1_dp8_tp1_ep4():
    """
    test case deepseek r1 w8a8 dp8 tp1 ep1
    """

    new_yaml = "dp8_tp1_ep4.yaml"
    replaced_pattern = ['data_parallel: 8', 'model_parallel: 1', 'expert_parallel: 4']
    dp_size = 8
    tp_size = 1
    # Sample prompts.
    prompts = [
        "You are a helpful assistant.<｜User｜>将文本分类为中性、负面或正面。 \n文本：我认为这次假期还可以。 "
        "\n情感：<｜Assistant｜>\n",
    ] * 8

    except_list = ['ugs611ాలు'] * 8
    exec_ds_with_dp(new_yaml, replaced_pattern, dp_size, tp_size, prompts, except_list)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.allcards
def test_deepseek_r1_tp8_ep8():
    """
    test case deepseek r1 w8a8 tp8 ep8
    """

    new_yaml = "tp8_ep8.yaml"
    replaced_pattern = ['data_parallel: 1', 'model_parallel: 8', 'expert_parallel: 8']
    # Sample prompts.
    prompts = [
        "You are a helpful assistant.<｜User｜>将文本分类为中性、负面或正面。 \n文本：我认为这次假期还可以。 "
        "\n情感：<｜Assistant｜>\n",
    ]

    except_list=['ugs611ాలు']
    exec_ds_without_dp(new_yaml, replaced_pattern, prompts, except_list)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.allcards
def test_deepseek_r1_tp8_ep4():
    """
    test case deepseek r1 w8a8 tp8 ep4
    """

    new_yaml = "tp8_ep4.yaml"
    replaced_pattern = ['data_parallel: 1', 'model_parallel: 8', 'expert_parallel: 4']
    # Sample prompts.
    prompts = [
        "You are a helpful assistant.<｜User｜>将文本分类为中性、负面或正面。 \n文本：我认为这次假期还可以。 "
        "\n情感：<｜Assistant｜>\n",
    ]

    except_list=['ugs611ాలు']
    exec_ds_without_dp(new_yaml, replaced_pattern, prompts, except_list)
