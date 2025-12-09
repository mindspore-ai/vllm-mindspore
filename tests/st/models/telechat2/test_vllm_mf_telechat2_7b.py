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
"""test mf telechat2 7b."""
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
    "MS_ENABLE_LCCL": "off",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
}


def run_mf_telechat2_7b_network():
    """Run telechat2_7b network and check result."""
    # isort: off
    import vllm_mindspore
    from vllm import LLM, SamplingParams
    # isort: on

    # Sample prompts.
    message = [{"role": "user", "content": "I love Beijing because: "}]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

    # Create an LLM.
    llm = LLM(model=MODEL_PATH["telechat2_7b"],
              gpu_memory_utilization=0.9,
              trust_remote_code=True,
              tensor_parallel_size=1)
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.chat(message, sampling_params)
    except_list = [' Beijing is a city that truly captivates with its']
    # Print the outputs.
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        assert generated_text == except_list[i]


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_mf_telechat2_7b():
    """
    Test Summary:
        Test mcore telechat2 model inference.
    Expected Result:
        Running successfully, the request result meets expectations.
    Model Info:
        telechat2_7b
    """
    run_mf_telechat2_7b_network()
