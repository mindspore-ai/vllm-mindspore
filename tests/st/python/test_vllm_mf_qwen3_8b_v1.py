#!/usr/bin/env python3
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
from similarity import compare_distance
import vllm_mindspore  # noqa: F401
from vllm import LLM, SamplingParams

env_manager = set_env.EnvVarManager()
# def env
env_vars = {
    "MINDFORMERS_MODEL_CONFIG": "./config/predict_qwen3_8b_instruct.yaml",
    "ASCEND_CUSTOM_PATH": os.path.expandvars("$ASCEND_HOME_PATH/../"),
    "vLLM_MODEL_BACKEND": "MindFormers",
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


class TestMfQwenV1:
    """
    Test Qwen.
    """

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_mf_qwen_batch(self):
        """
        test case qwen3 8B, to test prefill and decode mixed, can trigger PA q_seq_len > 1
        """
        # Sample prompts.
        prompts = [
            "北京烤鸭是",
            "请介绍一下华为，华为是",
            "今年似乎大模型之间的内卷已经有些偃旗息鼓了，各大技术公司逐渐聪单纯追求模型参数量的竞赛中抽身,"
            "转向更加注重模型的实际>应用效果和效率",
        ] * 2

        # Create a sampling params object.
        sampling_params = SamplingParams(temperature=0.0,
                                         max_tokens=128,
                                         top_k=1)

        # Create an LLM.
        llm = LLM(model="/home/workspace/mindspore_dataset/weight/Qwen3-8B",
                  block_size=32,
                  gpu_memory_utilization=0.9,
                  tensor_parallel_size=1)
        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        outputs = llm.generate(prompts, sampling_params)
        except_list = [
            "北京的著名菜系，其特点是皮脆肉嫩，色泽金黄，味道鲜美。北京烤鸭的历史可以追溯到明朝，"
            "当时它被作为宫廷菜肴，后来逐渐流传到民间。北京烤鸭的制作过程非常讲究，通常选用肥嫩的填鸭，"
            "经过特殊的腌制和烤制工艺，使其皮脆肉嫩。烤鸭的吃法也颇具特色，通常将烤鸭片成薄片，配以薄饼、"
            "甜面酱、葱丝、黄瓜条等配料，卷起来食用。北京烤鸭不仅是一道美食，更是一种文化象征，代表着",
            "做什么的，有哪些产品，有哪些子公司，有哪些业务领域，有哪些技术，有哪些专利，有哪些荣誉，"
            "有哪些合作伙伴，有哪些竞争对手，有哪些产品线，有哪些服务，有哪些市场，有哪些用户群体，"
            "有哪些创新，有哪些挑战，有哪些未来发展方向，有哪些社会责任，有哪些文化，有哪些历史，"
            "有哪些里程碑，有哪些产品特点，有哪些技术优势，有哪些产品优势，有哪些服务优势，有哪些市场优势，"
            "有哪些用户优势，有哪些创新优势，有哪些挑战优势，有哪些未来发展方向优势，有哪些社会责任优势，"
            "有哪些文化优势，有哪些历史优势，有哪些里程碑优势，有哪些产品特点优势，有哪些技术优势优势，有哪些",
            "。这种转变不仅体现在模型的参数量上,还体现在模型的训练和推理速度上。例如,Meta的Llama3-8B和"
            "Llama3-70B,以及阿里巴巴的Qwen2-72B,这些大模型在参数量上虽然依然庞大,但它们的训练和推理速度"
            "已经显著提升。此外,还有许多其他公司也在积极研发更高效、更实用的大模型,如Google的Gemini、"
            "Anthropic的Claude、OpenAI的GPT-4o等。这些模型不仅在参数量上有所突破,更在实际应用中展现出",
        ] * 2
        # Print the outputs.
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            compare_distance(generated_text, except_list[i], bench_sim=0.95)

        # unset env
        env_manager.unset_all()
