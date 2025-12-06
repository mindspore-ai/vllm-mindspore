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
"""test mf deepseek r1."""
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
    "VLLM_MS_MODEL_BACKEND": "MindFormers",
    "MS_ENABLE_LCCL": "on",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0"
}


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
def test_deepseek_r1_bf16():
    """
    Test Summary:
        Test case deepseek r1 bf16 model inference.
    Expected Result:
        Running successfully, the request result meets expectations.
    Model Info:
        DeepSeek-R1-bf16
    """
    import vllm_mindspore
    from vllm import LLM, SamplingParams
    # Sample prompts.
    prompts = [
        "You are a helpful assistant.<｜User｜>将文本分类为中性、负面或正面。"
        " \n文本：我认为这次假期还可以。 \n情感：<｜Assistant｜>\n",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10, top_k=1)

    # Create an LLM.
    llm = LLM(model=MODEL_PATH["DeepSeek-R1-bf16"],
              trust_remote_code=True,
              gpu_memory_utilization=0.9,
              tensor_parallel_size=2,
              max_model_len=33 * 1024)
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    except_list = ['ugs611ాలు sic辨hara的开璞 SquaresInsp']
    # Print the outputs.
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        assert generated_text == except_list[i]
