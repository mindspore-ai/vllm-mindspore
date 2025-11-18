# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologites Co., Ltd
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
"""test vllm qwen3."""
import pytest
from unittest.mock import patch

import os

from tests.st.python.utils.cases_parallel import cleanup_subprocesses
from tests.st.python.utils.env_var_manager import EnvVarManager


def teardown_function():
    cleanup_subprocesses()


env_manager = EnvVarManager()
env_manager.setup_mindformers_environment()
# def env
env_vars = {
    "ASCEND_CUSTOM_PATH": os.path.expandvars("$ASCEND_HOME_PATH/../"),
    "VLLM_MS_MODEL_BACKEND": "Native",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "MS_ALLOC_CONF": "enable_vmm:True",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
    "VLLM_USE_V1": "1",
}


def run_vllm_qwen3_8b(enforce_eager=False):
    """
    run case qwen3 8B
    """
    from vllm import LLM, SamplingParams

    # Sample prompts.
    prompts = [
        "<|im_start|>user\n将文本分类为中性、负面或正面。 "
        "\n文本：我认为这次假期还可以。 \n情感："
        "<|im_end|>\n<|im_start|>assistant\n",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10, top_k=1)

    # Create an LLM.
    llm = LLM(model="/home/workspace/mindspore_dataset/weight/Qwen3-8B",
              gpu_memory_utilization=0.9,
              tensor_parallel_size=1,
              enforce_eager=enforce_eager,
              max_model_len=4096)
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    except_list = ['<think>\n好的，我现在需要处理用户的查询，']
    # Print the outputs.
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        assert generated_text == except_list[
            i], f"Expected: {except_list[i]}, but got: {generated_text}"


@patch.dict(os.environ, env_vars)
def test_vllm_qwen3_0_6b():
    """
    test case qwen3 0.6B
    """
    import vllm_mindspore
    from vllm import LLM, SamplingParams

    # Sample prompts.
    prompts = [
        "<|im_start|>user\n将文本分类为中性、负面或正面。 "
        "\n文本：我认为这次假期还可以。 \n情感：<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10, top_k=1)

    # Create an LLM.
    llm = LLM(model="/home/workspace/mindspore_dataset/weight/Qwen3-0.6B",
              gpu_memory_utilization=0.9,
              tensor_parallel_size=1,
              max_model_len=4096)
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    except_list = ['情感：中性']
    # Print the outputs.
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        assert generated_text == except_list[
            i], f"Expected: {except_list[i]}, but got: {generated_text}"


@patch.dict(os.environ, env_vars)
def test_vllm_qwen3_8b():
    """
    test case qwen3 8B
    """
    import vllm_mindspore
    run_vllm_qwen3_8b()


@patch.dict(os.environ, env_vars)
def test_qwen3_enforce_eager():
    """
    Test qwen3 8B using ENFORCE_EAGER.
    """
    import vllm_mindspore
    run_vllm_qwen3_8b(enforce_eager=True)
