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
"""test vllm qwen3 moe."""
import os

from tests.st.python import utils


def teardown_function():
    utils.cleanup_subprocesses()


env_manager = utils.EnvVarManager()
# def env
env_vars = {
    "ASCEND_CUSTOM_PATH": os.path.expandvars("$ASCEND_HOME_PATH/../"),
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "VLLM_MS_MODEL_BACKEND": "Native",
    "MS_ALLOC_CONF": "enable_vmm:True",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
    "VLLM_USE_V1": "1",
}
# set env
env_manager.setup_ai_environment(env_vars)
import vllm_mindspore
from vllm import LLM, SamplingParams


def run_vllm_qwen3_30b_a3b(enforce_eager=False):
    """
    test case qwen3-30B-A3B
    """

    # Sample prompts.
    prompts = [
        "<|im_start|>user\n将文本分类为中性、负面或正面。 "
        "\n文本：我认为这次假期还可以。 \n情感："
        "<|im_end|>\n<|im_start|>assistant\n",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10, top_k=1)

    # Create an LLM.
    llm = LLM(
        model="/home/workspace/mindspore_dataset/weight/Qwen3-30B-A3B",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,
        max_model_len=4096,
        enforce_eager=enforce_eager,
    )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    except_list = ['<think>\n好的，我现在需要处理这个文本分类']
    # Print the outputs.
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        assert generated_text == except_list[
            i], f"Expected: {except_list[i]}, but got: {generated_text}"

    # unset env
    env_manager.unset_all()


def test_vllm_qwen3_30b_a3b():
    """
    test case qwen3-30B-A3B
    """

    run_vllm_qwen3_30b_a3b()


def test_vllm_qwen3_30b_a3b_eager():
    """
    test case qwen3-30B-A3B eager mode
    """

    run_vllm_qwen3_30b_a3b(enforce_eager=True)
