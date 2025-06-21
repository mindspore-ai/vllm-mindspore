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

# isort:skip_file
"""test vllm llama3."""
import os

import pytest

from tests.st.python import set_env

env_manager = set_env.EnvVarManager()
# def env
env_vars = {
    "ASCEND_CUSTOM_PATH": os.path.expandvars("$ASCEND_HOME_PATH/../"),
    "MS_ENABLE_LCCL": "off",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "MS_ALLOC_CONF": "enable_vmm:True",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
    "VLLM_USE_V1": "1",
    "HCCL_IF_BASE_PORT": "60000"
}
# set env
env_manager.setup_ai_environment(env_vars)
import vllm_mindspore
from vllm import LLM, SamplingParams


def test_vllm_llama3_8b():
    """
    test case llama3.1 8B
    """

    # Sample prompts.
    prompts = [
        "<|start_header_id|>user<|end_header_id|>\n\n将文本分类为中性、负面或正面。 "
        "\n文本：我认为这次假期还可以。 \n情感：<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10, top_k=1)

    # Create an LLM.
    llm = LLM(
        model="/home/workspace/mindspore_dataset/weight/Llama-3.1-8B-Instruct",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        max_model_len=4096)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    except_list = ['中性']
    # Print the outputs.
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        assert generated_text == except_list[i]

    # unset env
    env_manager.unset_all()


def test_vllm_llama3_1b():
    """
    test case llama3.2 1B
    """

    # Sample prompts.
    prompts = [
        "<|start_header_id|>user<|end_header_id|>\n\n将文本分类为中性、负面或正面。 "
        "\n文本：我认为这次假期还可以。 \n情感：<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10, top_k=1)

    # Create an LLM.
    llm = LLM(
        model="/home/workspace/mindspore_dataset/weight/Llama-3.2-1B-Instruct",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        max_model_len=4096)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    except_list = ['中性']
    # Print the outputs.
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        assert generated_text == except_list[i]

    # unset env
    env_manager.unset_all()
