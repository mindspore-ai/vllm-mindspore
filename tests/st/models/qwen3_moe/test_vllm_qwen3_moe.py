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
import pytest
from unittest.mock import patch

import os

from tests.utils.common_utils import (teardown_function, setup_function,
                                      MODEL_PATH)
from tests.utils.env_var_manager import EnvVarManager

env_manager = EnvVarManager()
env_manager.setup_mindformers_environment()
# def env
env_vars = {
    "VLLM_MS_MODEL_BACKEND": "Native",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
}


def run_vllm_qwen3_30b_a3b(enforce_eager=False,
                           tp_size=2,
                           enable_aclgraph=False):
    """
    test case qwen3-30B-A3B
    """
    import vllm_mindspore
    from vllm import LLM, SamplingParams

    # Sample prompts.
    prompts = [
        "<|im_start|>user\n将文本分类为中性、负面或正面。 "
        "\n文本：我认为这次假期还可以。 \n情感："
        "<|im_end|>\n<|im_start|>assistant\n",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10, top_k=1)

    compilation_level = 0
    if enable_aclgraph:
        compilation_level = 3

    # Create an LLM.
    llm = LLM(
        model=MODEL_PATH["Qwen3-30B-A3B"],
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tp_size,
        max_model_len=4096,
        enforce_eager=enforce_eager,
        compilation_config=compilation_level,
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


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
def test_vllm_qwen3_30b_a3b():
    """
    Test Summary:
        Test native qwen3 moe model inference.
    Expected Result:
        Running successfully, the request result meets expectations.
    Model Info:
        Qwen3-30B-A3B
    """
    """
    test case qwen3-30B-A3B
    """

    run_vllm_qwen3_30b_a3b()


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
def test_vllm_qwen3_30b_a3b_eager():
    """
    Test Summary:
        Test native qwen3 moe model inference in eager mode.
    Expected Result:
        Running successfully, the request result meets expectations.
    Model Info:
        Qwen3-30B-A3B
    """

    run_vllm_qwen3_30b_a3b(enforce_eager=True)


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
def test_vllm_qwen3_30b_a3b_310p():
    """
    Test Summary:
        Test native qwen3 moe model inference in eager mode.
        4 cards are required to execute on 310P.
    Expected Result:
        Running successfully, the request result meets expectations.
    Model Info:
        Qwen3-30B-A3B
    """

    run_vllm_qwen3_30b_a3b(tp_size=4)


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
def test_vllm_qwen3_30b_a3b_aclgraph():
    """
    Test Summary:
        test case qwen3 30B with aclgraph in two devices
    Expected Result:
        Running successfully, the request result meets expectations.
    Model Info:
        Qwen3-30B
    """

    run_vllm_qwen3_30b_a3b(enforce_eager=False,
                           tp_size=2,
                           enable_aclgraph=True)
