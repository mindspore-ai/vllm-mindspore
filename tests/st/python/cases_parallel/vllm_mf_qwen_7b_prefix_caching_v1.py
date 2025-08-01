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

# type: ignore
# isort: skip_file

import os
from tests.st.python import utils


def teardown_function():
    utils.cleanup_subprocesses()


env_manager = utils.EnvVarManager()
env_vars = {
    "MINDFORMERS_MODEL_CONFIG": "./config/predict_qwen2_5_7b_instruct.yaml",
    "ASCEND_CUSTOM_PATH": os.path.expandvars("$ASCEND_HOME_PATH/../"),
    "vLLM_MODEL_BACKEND": "MindFormers",
    "MS_ENABLE_LCCL": "off",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0"
}
env_manager.setup_ai_environment(env_vars)
import vllm_mindspore
from vllm import LLM, SamplingParams


def test_mf_qwen_7b_prefix_caching():
    """
    test case qwen_7b_prefix_caching
    """

    # First prompts.
    prompts = [
        "I love Beijing, because it is a city that has so much to offer. I have visited"
    ]
    # second prompts, the second prompt is a continuation of the first prompts, make sure prefix caching work.
    second_prompts = [
        "I love Beijing, because it is a city that has so much to offer. I have visited many places"
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10, top_k=1)

    # Create an LLM.
    llm = LLM(
        model="/home/workspace/mindspore_dataset/weight/Qwen2.5-7B-Instruct",
        max_model_len=8192,
        block_size=16,
        tensor_parallel_size=2)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    second_outputs = llm.generate(second_prompts, sampling_params)
    except_list = [' many times and each time I have found something new']
    second_except_list = [' in Beijing, but I have to say that the']
    for i, (output, second_output) in enumerate(zip(outputs, second_outputs)):
        generated_text = output.outputs[i].text
        print(
            f"Output1 - Prompt: {prompts[i]!r}, Generated text: {generated_text!r}"
        )
        assert generated_text == except_list[i]

        second_generated_text = second_output.outputs[i].text
        print(
            f"Output2 - Prompt: {second_prompts[i]!r}, Generated text: {second_generated_text!r}"
        )
        assert second_generated_text == second_except_list[i]

    env_manager.unset_all()
