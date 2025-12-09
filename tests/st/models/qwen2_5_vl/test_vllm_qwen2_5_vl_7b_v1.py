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
"""test mf qwen2.5 vl 7B."""
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

env_manager = EnvVarManager()
env_manager.setup_mindformers_environment()
# def env
env_vars = {
    "VLLM_MS_MODEL_BACKEND": "Native",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
}

PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
    "\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "What is in the image?<|im_end|>\n"
    "<|im_start|>assistant\n")

PROMPT_TEMPLATE_2 = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
    "\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "Is there anyone in the picture?<|im_end|>\n"
    "<|im_start|>assistant\n")
""" 
The test sample can be obtained through the following link: 
https://tools.mindspore.cn/dataset/workspace/mindspore_dataset/video_file/ 
korean_eating.mp4 
https://tools.mindspore.cn/dataset/workspace/mindspore_dataset/images/ 
houses_and_mountain.jpeg 
"""
video_path = "/home/workspace/mindspore_dataset/video_file/korean_eating.mp4"
image_path = \
    "/home/workspace/mindspore_dataset/images/houses_and_mountain.jpeg"
model_path = MODEL_PATH["Qwen2.5-VL-7B-Instruct"]


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
              max_num_batched_tokens=32)

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
        {
            "prompt": PROMPT_TEMPLATE_2,
            "multi_modal_data": {
                "image": pil_image()
            },
        },
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128, top_k=1)
    expect_list = [
        'The image depicts a serene and picturesque landscape. It features a '
        'lush green meadow with wildflowers in the foreground. In the middle '
        'ground, there are small wooden huts, possibly used for storage or as '
        'simple shelters. Beyond the meadow, there is a calm body of water, '
        'likely a lake, surrounded by dense forests. In the background, '
        'majestic mountains rise, their peaks partially covered with snow, '
        'suggesting a high-altitude location. The sky is partly cloudy, '
        'with soft lighting that enhances the tranquil and idyllic '
        'atmosphere of the scene. This type of landscape is often '
        'associated with alpine regions.'
    ]

    for i in range(3):
        # Generate texts from the prompts. The output is a list of
        # RequestOutput objects that contain the prompt,
        # generated text, and other information.
        outputs = llm.generate(inputs, sampling_params)
        # Print the outputs.
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            print(
                f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}"
            )
            compare_distance(generated_text, expect_list[0], bench_sim=0.95)


# read video frames related
def video_to_ndarrays(path: str, num_frames: int = -1) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {path!r}")
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    total = len(all_frames)
    if total == 0:
        raise ValueError(f"No frames found in video {path!r}")
    if num_frames > 0:
        if num_frames > total:
            raise ValueError(f"Requested {num_frames} "
                             f"frames but only {total} available")
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        all_frames = [all_frames[i] for i in indices]

    return np.stack(all_frames)


def prepare_text(processor, prompt: str):
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
            },
            {
                "type": "text",
                "text": f"{prompt}"
            },
        ],
    }]
    text = processor.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True)
    return text


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
def test_qwen2_5_vl_7b_v1():
    """
    Test Summary:
        Test native backend qwen2_5_vl model inference, with image inputs.
    Expected Result:
        Running successfully, the request result meets expectations.
    Model Info:
        Qwen2.5-VL-7B-Instruct
    """
    import vllm_mindspore

    llm = generate_llm_engine(enforce_eager=False, tensor_parallel_size=2)
    forward_and_check(llm)


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_qwen2_5_vl_7b_v1_enforce_eager():
    """
    Test Summary:
        Test native backend qwen2_5_vl model inference in eager mode,
        with image inputs.
    Expected Result:
        Running successfully, the request result meets expectations.
    Model Info:
        Qwen2.5-VL-7B-Instruct
    """
    import vllm_mindspore

    llm = generate_llm_engine(enforce_eager=True, tensor_parallel_size=1)
    forward_and_check(llm)


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_qwen2_5_vl_7b_v1_video_infer():
    """
    Test Summary:
        Test native backend qwen2_5_vl model inference, with video inputs.
    Expected Result:
        Running successfully, the request result meets expectations.
    Model Info:
        Qwen2.5-VL-7B-Instruct
    """
    import vllm_mindspore
    from vllm import LLM, SamplingParams

    frames = video_to_ndarrays(video_path, num_frames=10)
    print("frames shape", frames.shape)

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path)

    prompts = [
        "Describe this photo.", "Guess where and when this photo was taken."
    ]
    expect_list = [
        "The photo shows a group of people sitting around a table, engaged "
        "in an activity that involves blue food. The table is set with "
        "various bowls and utensils, and the individuals are wearing gloves,"
        " suggesting they are handling the food carefully. "
        "The setting appears to be indoors, possibly in a kitchen or a "
        "similar environment. The atmosphere seems to be casual and social, "
        "with the participants focused on their task.",
        "Based on the image, it appears to be taken in a casual dining "
        "setting, possibly a restaurant or a food court. The presence "
        "of a hot pot and the overall ambiance suggest it could be a "
        "place where people gather to enjoy a meal together. The text "
        "in the image is in Korean, which indicates that the photo was "
        "likely taken in a Korean-speaking region. The specific location "
        "and exact time cannot be determined from the image alone."
    ]

    texts = [prepare_text(processor, prompt) for prompt in prompts]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512, top_k=1)

    # Create an LLM.
    llm = LLM(
        model=model_path,
        limit_mm_per_prompt={'video': 1},
        max_model_len=32768,
    )

    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    inputs = [{
        "prompt": texts[i],
        "multi_modal_data": {
            "video": [frames]
        }
    } for i in range(len(texts))]
    outputs = llm.generate(inputs, sampling_params)

    # Print the outputs.
    for prompt, output, expect in zip(prompts, outputs, expect_list):
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        compare_distance(generated_text, expect, bench_sim=0.95)
