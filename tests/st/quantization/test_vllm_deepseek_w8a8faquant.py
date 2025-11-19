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
"""
test mf deepseek r1 w8a8faquant.
isort:skip_file
"""

# type: ignore
# isort: skip_file
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


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
def test_deepseek_r1():
    """
    Test Summary:
        test case for deepseek r1 w8a8faquant from golden-stick.
    Expected Result:
        Running successfully, the request result meets expectations.
    Model Info:
        DeepSeek-R1-w8a8faquant, the details see model_info.yaml.
    """
    import vllm_mindspore
    from vllm import LLM, SamplingParams

    # Sample prompts.
    prompts = [
        "介绍下北京故宫",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=5, top_k=1)

    # Create an LLM.
    llm = LLM(model=MODEL_PATH["DeepSeek-R1-W8A8FA-mcore"],
              trust_remote_code=True,
              gpu_memory_utilization=0.4,
              tensor_parallel_size=2,
              max_model_len=4096,
              quantization='ascend',
              kv_cache_dtype='int8')
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        assert "博物院ODాలు SER비스" in generated_text
