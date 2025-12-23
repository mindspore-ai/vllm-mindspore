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
"""test vllm minicpm4-0.5b."""
import pytest
from unittest.mock import patch

import os

from tests.utils.common_utils import (teardown_function, setup_function,
                                      MODEL_PATH)

# def env
env_vars = {
    "VLLM_MS_MODEL_BACKEND": "Native",
    "MS_ENABLE_LCCL": "off",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
}


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_vllm_minicpm4_0_5b():
    """
    Test Summary:
        Test case MiniCPM4-0.5B model inference.
    Expected Result:
        Running successfully, the request result meets expectations.
    Model Info:
        MiniCPM4-0.5B
    """
    import vllm_mindspore
    from vllm import LLM, SamplingParams

    # Sample prompts.
    prompts = [
        "<|im_start|>user\n请直接回答，中国的首都是那座城市？<|im_end|>\n<|im_start|>assistant\n",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10, top_k=1)

    # Create an LLM.
    llm = LLM(model=MODEL_PATH["MiniCPM4-0.5B"],
              gpu_memory_utilization=0.9,
              tensor_parallel_size=1,
              max_model_len=4096,
              trust_remote_code=True)
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    expected_list = ['中国的首都是北京。']
    # Print the outputs.
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        assert generated_text == expected_list[i]
