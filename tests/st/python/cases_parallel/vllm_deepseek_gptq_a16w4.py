#!/usr/bin/env python3
# isort: skip_file
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test mf deepseek r1 gptq int4 quantization."""
import os
import yaml
import pytest
from tests.st.python import set_env

env_manager = set_env.EnvVarManager()
# def env
env_vars = {
    "MINDFORMERS_MODEL_CONFIG": "./config/predict_deepseek_r1_671b_a16w4.yaml",
    "ASCEND_CUSTOM_PATH": os.path.expandvars("$ASCEND_HOME_PATH/../"),
    "vLLM_MODEL_BACKEND": "MindFormers",
    "MS_ENABLE_LCCL": "off",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "MS_ALLOC_CONF": "enable_vmm:True",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
    "VLLM_USE_V1": "0"
}
# set env
env_manager.setup_ai_environment(env_vars)
import vllm_mindspore  # noqa: F401, E402
from vllm import LLM, SamplingParams  # noqa: E402


def test_deepseek_r1_gptq_a16w4():
    """
    test case deepseek r1 a16w4
    """
    yaml_path = "./config/predict_deepseek_r1_671b.yaml"
    a16w4_yaml = "./config/predict_deepseek_r1_671b_a16w4.yaml"
    with open(yaml_path, 'r', encoding='utf-8') as file:
        content = yaml.safe_load(file)
    model_config = content["model"]["model_config"]
    model_config["quantization_config"] = {"quant_method": "gptq-pergroup"}
    content["model"]["model_config"] = model_config

    with open(a16w4_yaml, 'w', encoding='utf-8') as file:
        yaml.dump(content, file, allow_unicode=True, sort_keys=False)

    # Sample prompts.
    prompts = [
        "介绍下北京故宫",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024, top_k=1)

    # Create an LLM.
    llm = LLM(
        model=
        "/home/workspace/mindspore_dataset/weight/DeepSeekR1_gptq-pergroup_safetensors",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=4,
        max_model_len=4096)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        assert "博物院christianాలు sic辨" in generated_text

    # unset env
    env_manager.unset_all()
