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
"""test mf qwen3."""
import os

from tests.st.python import utils


def teardown_function():
    utils.cleanup_subprocesses()


env_manager = utils.EnvVarManager()
# def env
env_vars = {
    "ASCEND_CUSTOM_PATH": os.path.expandvars("$ASCEND_HOME_PATH/../"),
    "vLLM_MODEL_BACKEND": "MindFormers",
    "MS_ENABLE_LCCL": "off",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "MS_ALLOC_CONF": "enable_vmm:True",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
}


def run_mf_qwen3_networt():
    """Run qwen3 network and check result."""
    # isort: off
    import vllm_mindspore
    from vllm import LLM, SamplingParams
    # isort: on

    # Sample prompts.
    prompts = [
        "You are a helpful assistant.<｜User｜>将文本分类为中性、负面或正面。 \n"
        "文本：我认为这次假期还可以。 \n情感：<｜Assistant｜>\n",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10, top_k=1)

    # Create an LLM.
    llm = LLM(model="/home/workspace/mindspore_dataset/weight/Qwen3-8B",
              gpu_memory_utilization=0.9,
              tensor_parallel_size=2)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    except_list = ['好的，我需要分析用户提供的文本“我认为']
    # Print the outputs.
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        assert generated_text == except_list[i]


def test_mf_qwen3_v0():
    """Test qwen3 8B using V0 LLMEngine."""
    env_vars["VLLM_USE_V1"] = "0"
    env_manager.setup_ai_environment(env_vars)
    run_mf_qwen3_networt()
    env_manager.unset_all()


def test_mf_qwen3_v1():
    """Test qwen3 8B using V0 LLMEngine."""
    env_vars["VLLM_USE_V1"] = "1"
    env_manager.setup_ai_environment(env_vars)
    run_mf_qwen3_networt()
    env_manager.unset_all()
