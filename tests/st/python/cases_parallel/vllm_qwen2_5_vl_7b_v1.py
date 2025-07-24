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
import os

from PIL import Image
from tests.st.python import utils
from tests.st.python.cases_parallel.similarity import compare_distance


def teardown_function():
    utils.cleanup_subprocesses()


env_manager = utils.EnvVarManager()
# def env
env_vars = {
    "ASCEND_CUSTOM_PATH": os.path.expandvars("$ASCEND_HOME_PATH/../"),
    "HCCL_OP_EXPANSION_MODE": "AIV",
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

PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
    "\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "What is in the image?<|im_end|>\n"
    "<|im_start|>assistant\n")


def pil_image() -> Image.Image:
    image_path = "images/1080p.jpeg"
    return Image.open(image_path)


def test_qwen2_5_vl_7b_v1():
    """
    test case qwen2.5 vl 7B
    """
    inputs = [{
        "prompt": PROMPT_TEMPLATE,
        "multi_modal_data": {
            "image": pil_image()
        },
    }]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128, top_k=1)

    # Create an LLM.
    llm = LLM(
        model="/home/workspace/mindspore_dataset/weight/Qwen2.5-VL-7B-Instruct",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,
        max_model_len=4096,
        max_num_seqs=32,
        max_num_batched_tokens=32)
    except_list = [
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
            compare_distance(generated_text, except_list[0], bench_sim=0.95)

    # unset env
    env_manager.unset_all()
