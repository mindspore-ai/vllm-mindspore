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

from tests.utils.common_utils import (teardown_function, setup_function,
                                      MODEL_PATH)

# def env
env_vars = {
    "VLLM_MS_MODEL_BACKEND": "Native",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
}


def run_vllm_qwen3_8b(enforce_eager=False, tp_size=1):
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
    llm = LLM(model=MODEL_PATH["Qwen3-8B"],
              gpu_memory_utilization=0.9,
              tensor_parallel_size=tp_size,
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
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_vllm_qwen3_0_6b():
    """
    Test Summary:
        test case qwen3 0.6B
    Expected Result:
        Running successfully, the request result meets expectations.
    Model Info:
        Qwen3-8B
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
    llm = LLM(model=MODEL_PATH["Qwen3-0.6B"],
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
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_vllm_qwen3_8b():
    """
    Test Summary:
        Test qwen3 8B with graph mode.
    Expected Result:
        Running successfully, the request result meets expectations
    Model Info:
        Qwen3-8B
    """
    import vllm_mindspore
    run_vllm_qwen3_8b()


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_qwen3_enforce_eager():
    """
    Test Summary:
        Test qwen3 8B using enforce_eager.
    Expected Result:
        Running successfully, the request result meets expectations
    Model Info:
        Qwen3-8B
    """
    import vllm_mindspore
    run_vllm_qwen3_8b(enforce_eager=True)


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
def test_vllm_qwen3_8b_310p():
    """
    Test Summary:
        Test qwen3 8B using enforce_eager.
    Expected Result:
        Running successfully, the request result meets expectations
    Model Info:
        Qwen3-8B
    """
    import vllm_mindspore
    run_vllm_qwen3_8b(tp_size=2)


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_vllm_qwen3_0_6b_aclgraph():
    """
    Test Summary:
        test case qwen3 0.6B with aclgraph
    Expected Result:
        Running successfully, the request result meets expectations.
    Model Info:
        Qwen3-8B
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
    llm = LLM(model=MODEL_PATH["Qwen3-0.6B"],
              gpu_memory_utilization=0.9,
              tensor_parallel_size=1,
              max_model_len=4096,
              compilation_config=3)
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
