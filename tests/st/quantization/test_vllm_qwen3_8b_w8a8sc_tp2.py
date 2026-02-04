# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Huawei Technologies Co., Ltd.
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
test native qwen3 8b w8a8sc sparse quantization with TP2.
"""

# type: ignore
# isort: skip_file
import pytest
from unittest.mock import patch

import os
from pathlib import Path
from tests.utils.common_utils import (teardown_function, setup_function,
                                      MODEL_PATH)

# def env
env_vars = {
    "VLLM_MS_MODEL_BACKEND": "Native",
    "MS_ENABLE_LCCL": "off",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
}


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
def test_qwen3_8b_w8a8sc_tp2():
    """
    Test Summary:
        Test case for Qwen3-8B with W8A8SC sparse quantization from
        golden-stick, using tensor parallelism size 2.
    Expected Result:
        Running successfully, the request result meets expectations.
    Model Info:
        Qwen3-8B-W8A8SC-TP2, the details see model_info.yaml.
    """
    import vllm_mindspore
    from vllm import LLM, SamplingParams

    # Sample prompts.
    prompts = [
        "<|im_start|>user\n将文本分类为中性、负面或正面。 "
        "\n文本：我认为这次假期还可以。 \n情感："
        "<|im_end|>\n<|im_start|>assistant\n",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1000, top_k=1)

    # Create an LLM.
    # For sparse quantization models, the model path should point to the parent
    # directory containing rank_0, rank_1 subdirectories (for TP2).
    # Each rank directory contains safetensors files for that rank.
    # Note: As a result the structure of sparse quantization model,
    # load_format must be explicitly set to 'sparse_quant' or 'auto'
    # for sparse quantization models to use the correct model loader.
    llm = LLM(model=MODEL_PATH["Qwen3-8B-W8A8SC-TP2"],
              trust_remote_code=True,
              gpu_memory_utilization=0.3,
              tensor_parallel_size=2,
              max_model_len=4096,
              quantization='golden-stick',
              load_format='sparse_quant')
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        # Assert that generated text is not empty
        assert len(generated_text) > 0, "Generated text should not be empty"
