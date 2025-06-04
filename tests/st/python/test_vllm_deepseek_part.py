#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
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
# ============================================================================
"""test mf deepseek r1."""
import pytest
import os
from . import set_env
env_manager = set_env.EnvVarManager()
# def env
env_vars = {
    "MINDFORMERS_MODEL_CONFIG": "./config/predict_deepseek_r1_671b_w8a8.yaml",
    "ASCEND_CUSTOM_PATH": os.path.expandvars("$ASCEND_HOME_PATH/../"),
    "vLLM_MODEL_BACKEND": "MindFormers",
    "MS_ENABLE_LCCL": "on",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    "MS_ALLOC_CONF": "enable_vmm:True",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
    "VLLM_USE_V1": "0",
}
# set env
env_manager.setup_ai_environment(env_vars)
import vllm_mindspore
from vllm import LLM, SamplingParams

class TestDeepSeek:
    """
    Test Deepseek.
    """

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.skip(reason="gs master branch is not suit for the newest mindformers.")
    def test_deepseek_r1(self):
        """
        test case deepseek r1 w8a8
        """

        # Sample prompts.
        prompts = [
            "You are a helpful assistant.<｜User｜>将文本分类为中性、负面或正面。 \n文本：我认为这次假期还可以。 \n情感：<｜Assistant｜>\n",
        ]

        # Create a sampling params object.
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10, top_k=1)

        # Create an LLM.
        llm = LLM(model="/home/workspace/mindspore_dataset/weight/DeepSeek-R1-W8A8",
                  trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=8)
        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        outputs = llm.generate(prompts, sampling_params)
        except_list=['ugs611ాలు哒ాలు mahassisemaSTE的道德', 'ugs611ాలు哒ాలు mah战区rollerOVERlaid']
        # Print the outputs.
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            assert generated_text in except_list

        # unset env
        env_manager.unset_all()


class TestDeepSeekMTP:
    """
    Test DeepseekMTP.
    大模型用量化（4层），mtp模型用浮点（1层，layer 61）。
    mtp的权重加载默认从配置的num_hidden_layer开始，为了支持减层推理场景mtp权重加载，ci服务器上修改了浮点的权重map文件的layer为4。
    """
    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.skip(reason="gs master branch is not suit for the newest mindformers.")
    def test_deepseek_mtp(self):
        """
        test case deepseek mtp with main model of r1-w8a8
        """

        # Sample prompts.
        prompts = [
            "You are a helpful assistant.<｜User｜>将文本分类为中性、负面或正面。 \n文本：我认为这次假期还可以。 \n情感：<｜Assistant｜>\n",
        ]

        # Create a sampling params object.
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10, top_k=1)

        # Create an LLM.
        llm = LLM(model="/home/workspace/mindspore_dataset/weight/DeepSeek-R1-MTP",
                  trust_remote_code=True, gpu_memory_utilization=0.7, tensor_parallel_size=8,
                  speculative_config={"num_speculative_tokens":1})
        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        outputs = llm.generate(prompts, sampling_params)
        except_list = ['ugs611ాలు哒ాలు mahassisemaSTE的道德', 'ugs611ాలు哒ాలు mah战区rollerOVERlaid']
        # Print the outputs.
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            assert generated_text in except_list

        # unset env
        env_manager.unset_all()
