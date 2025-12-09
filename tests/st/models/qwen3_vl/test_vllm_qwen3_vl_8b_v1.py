# SPDX-License-Identifier: Apache-2.0

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
"""test native qwen3 vl 8B."""
import pytest
from unittest.mock import patch

import os

import cv2
import numpy as np
from PIL import Image
from tests.utils.env_var_manager import EnvVarManager
from tests.st.models.qwen2_5_vl.similarity import compare_distance
from tests.utils.common_utils import (teardown_function, setup_function,
                                      MODEL_PATH)

# def env
env_vars = {
    "VLLM_MS_MODEL_BACKEND": "Native",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
}

PROMPT_TEMPLATE = (
    "<|im_start|>user\nWhat is in the image?<|vision_start|><|image_pad|>"
    "<|vision_end|><|im_end|>\n<|im_start|>assistant\n")

image_path = \
    "/home/workspace/mindspore_dataset/images/houses_and_mountain.jpeg"
model_path = MODEL_PATH["Qwen3-VL-8B-Instruct"]


def pil_image() -> Image.Image:
    return Image.open(image_path)


def generate_llm_engine(enforce_eager=False, tensor_parallel_size=1):
    from vllm import LLM
    # Create an LLM.
    llm = LLM(model=model_path,
              gpu_memory_utilization=0.9,
              tensor_parallel_size=tensor_parallel_size,
              enforce_eager=enforce_eager,
              max_model_len=4096,
              max_num_seqs=32,
              max_num_batched_tokens=4096,
              limit_mm_per_prompt={"video": 0})

    return llm


def forward_and_check(llm):
    from vllm import SamplingParams
    inputs = [
        {
            "prompt": PROMPT_TEMPLATE,
            "multi_modal_data": {
                "image": pil_image()
            },
        },
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128, top_k=1)
    expect_list = [
        "This is a beautiful, scenic landscape photograph. Here's a "
        "breakdown of what's in the image:\n\n* Foreground: A vibrant green "
        "meadow dotted with wildflowers, including yellow daisies and pink "
        "blossoms. A small, rustic wooden shed with a tiled roof sits "
        "prominently in the grass.\n* Midground: A calm, blue lake or alpine "
        "tarn stretches across the scene, surrounded by more green fields "
        "and a few other small wooden structures. The shoreline is lined "
        "with dense forests of evergreen trees.\n* Background: A majestic "
        "range of snow-capped mountains rises in the distance, their peaks "
        "catching the"
    ]

    outputs = llm.generate(inputs, sampling_params)
    # Print the outputs.
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")
        compare_distance(generated_text, expect_list[0], bench_sim=0.95)


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
def test_qwen3_vl_8b_v1():
    """
    Test Summary:
        Test native qwen3 vl model inference.
    Expected Result:
        Running successfully, the request result meets expectations.
    Model Info:
        Qwen3-VL-8B-Instruct
    """
    import vllm_mindspore

    llm = generate_llm_engine(enforce_eager=False, tensor_parallel_size=2)
    forward_and_check(llm)


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_qwen3_vl_8b_v1_enforce_eager():
    """
    Test Summary:
        Test native qwen3 vl model inference in eager mode.
    Expected Result:
        Running successfully, the request result meets expectations.
    Model Info:
        Qwen3-VL-8B-Instruct
    """
    import vllm_mindspore

    llm = generate_llm_engine(enforce_eager=True, tensor_parallel_size=1)
    forward_and_check(llm)
