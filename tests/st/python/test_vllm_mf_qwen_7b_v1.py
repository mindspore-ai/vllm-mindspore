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
"""test mf qwen."""
import pytest
import os
from . import set_env
env_manager = set_env.EnvVarManager()
# def env
env_vars = {
    "MINDFORMERS_MODEL_CONFIG": "./config/predict_qwen2_5_7b_instruct.yaml",
    "ASCEND_CUSTOM_PATH": os.path.expandvars("$ASCEND_HOME_PATH/../"),
    "vLLM_MODEL_BACKEND": "MindFormers",
    "MS_ENABLE_LCCL": "off",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ASCEND_RT_VISIBLE_DEVICES": "0,1",
    "MS_ALLOC_CONF": "enable_vmm:True",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
}
# set env
env_manager.setup_ai_environment(env_vars)
import vllm_mindspore
from vllm import LLM, SamplingParams


class TestMfQwenV1:
    """
    Test Qwen.
    """
    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_mf_qwen(self):
        """
        test case qwen2.5 7B
        """

        # Sample prompts.
        prompts = [
            "You are a helpful assistant.<｜User｜>将文本分类为中性、负面或正面。 \n文本：我认为这次假期还可以。 \n情感：<｜Assistant｜>\n",
        ]

        # Create a sampling params object.
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10, top_k=1)

        # Create an LLM.
        llm = LLM(model="/home/workspace/mindspore_dataset/weight/Qwen2.5-7B-Instruct",
                  gpu_memory_utilization=0.9, tensor_parallel_size=2)
        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        outputs = llm.generate(prompts, sampling_params)
        except_list=['中性<｜Assistant｜> 这句话']
        # Print the outputs.
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            assert generated_text == except_list[i]

        # unset env
        env_manager.unset_all()

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.skip(reason="pc precision need to be fixed on v0.8.3 V0")
    def test_mf_qwen_batch(self):
        """
        test case qwen2.5 7B, to test prefill and decode mixed, can trigger PA q_seq_len > 1
        """
        # Sample prompts.
        prompts = [
            "北京烤鸭是",
            "请介绍一下华为，华为是",
            "今年似乎大模型之间的内卷已经有些偃旗息鼓了，各大技术公司逐渐聪单纯追求模型参数量的竞赛中抽身,"
            "转向更加注重模型的实际>应用效果和效率",
        ] * 2

        # Create a sampling params object.
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10, top_k=1)

        # Create an LLM.
        llm = LLM(model="/home/workspace/mindspore_dataset/weight/Qwen2.5-7B-Instruct", block_size=32,
                  gpu_memory_utilization=0.9, tensor_parallel_size=2)
        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        outputs = llm.generate(prompts, sampling_params)
        except_list=[
            "享誉世界的中华美食，其制作工艺独特，",
            "做什么的？ 华为是一家中国公司，",
            "。 \n在这一背景下，阿里云发布了通",
        ] * 2
        # Print the outputs.
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            assert generated_text in except_list[i]

        # unset env
        env_manager.unset_all()

