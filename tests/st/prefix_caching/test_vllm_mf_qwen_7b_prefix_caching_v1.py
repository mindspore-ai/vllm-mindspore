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
"""test mf qwen prefix caching."""
import pytest
from unittest.mock import patch

import os
from tests.utils.common_utils import (teardown_function, setup_function,
                                      MODEL_PATH)
from tests.utils.env_var_manager import EnvVarManager

env_manager = EnvVarManager()
env_manager.setup_mindformers_environment()
env_vars = {
    "VLLM_MS_MODEL_BACKEND": "MindFormers",
    "MS_ENABLE_LCCL": "off",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0"
}


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
def test_mf_qwen_7b_prefix_caching():
    """
    Test Summary:
        For the online mindformers backend scenario with default mode,
        and then send two consecutive 2bs requests has same prefix.
    Expected Result:
        Successful execution with inference results meeting expectations.
    Model Info:
        Qwen2.5-7B-Instruct
    """
    import vllm_mindspore
    from vllm import LLM, SamplingParams

    # First prompts.
    prompts = [
        "I love Beijing, because it is a city that has so much to offer."
        " I have visited"
    ]
    # second prompts, the second prompt is a continuation of the first prompts,
    # make sure prefix caching work.
    second_prompts = [
        "I love Beijing, because it is a city that has so much to offer."
        " I have visited many places"
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10, top_k=1)

    # Create an LLM.
    llm = LLM(model=MODEL_PATH["Qwen2.5-7B-Instruct"],
              max_model_len=8192,
              block_size=16,
              tensor_parallel_size=2)
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    second_outputs = llm.generate(second_prompts, sampling_params)
    except_list = [' many times and each time I have found something new']
    second_except_list = [' in Beijing, but I have to say that the']
    for i, (output, second_output) in enumerate(zip(outputs, second_outputs)):
        generated_text = output.outputs[i].text
        print(f"Output1 - Prompt: {prompts[i]!r}, "
              f"Generated text: {generated_text!r}")
        assert generated_text == except_list[i]

        second_generated_text = second_output.outputs[i].text
        print(f"Output2 - Prompt: {second_prompts[i]!r}, "
              f"Generated text: {second_generated_text!r}")
        assert second_generated_text == second_except_list[i]
