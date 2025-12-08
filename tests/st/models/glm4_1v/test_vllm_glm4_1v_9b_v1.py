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
"""test mf qwen3 vl 8B."""
import pytest
from unittest.mock import patch

import os

import cv2
import numpy as np
from PIL import Image
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
    "[gMASK]<sop><|user|>\nWhat is in the image?<|begine_of_image|><|image|>"
    "<|end_of_image|><|assistant|>\n")

image_path = \
    "/home/workspace/mindspore_dataset/images/houses_and_mountain.jpeg"
model_path = MODEL_PATH["GLM-4.1V-9B-Thinking"]


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
        "Got it, let's analyze the image. The image shows a scenic landscape "
        "with several elements. First, there's a grassy field with colorful "
        "flowers (yellow and pink). Then, there are wooden cabins or small "
        "houses scattered in the meadow. In the middle, there's a lake or "
        "pond with blue water. Surrounding the area are dense forests with "
        "tall trees, and in the background, there are majestic mountains "
        "with some snow on their peaks, under a partly cloudy sky. So, the "
        "main elements are natural landscapes: meadows, flowers, cabins, a "
        "lake, forests, and mountains. I need to list what's"
    ]

    outputs = llm.generate(inputs, sampling_params)
    # Print the outputs.
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")
        compare_distance(generated_text, expect_list[0], bench_sim=0.95)


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
def test_glm4_1v_9b_v1():
    """
    test case glm4_1v 9B
    """
    import vllm_mindspore

    llm = generate_llm_engine(enforce_eager=False, tensor_parallel_size=2)
    forward_and_check(llm)


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_glm4_1v_9b_v1_enforce_eager():
    """
    test case glm4_1v 9B with eager mode
    """
    import vllm_mindspore

    llm = generate_llm_engine(enforce_eager=True, tensor_parallel_size=1)
    forward_and_check(llm)
