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
"""Example for generate with multimodal."""

import vllm_mindspore  # noqa: F401, E402

from PIL import Image
from vllm import LLM, SamplingParams  # noqa: E402


# Qwen2.5-VL
def get_llm(model_path: str, question: str, modality: str):
    llm = LLM(
        model=model_path,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        disable_mm_preprocessor_cache=True,
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    stop_token_ids = None

    return llm, prompt, stop_token_ids


def main(args):
    # Prepare args and inputs.
    img_question = "What is the content of this image?"
    img = Image.open("./imgs/1.jpg").convert("RGB")
    llm, prompt, stop_token_ids = get_llm(
        args.model_path, img_question, "image"
    )

    inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {"image": img},
        }
        for _ in range(2)
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.95,
        max_tokens=1024,
        stop_token_ids=stop_token_ids,
    )

    # Run generate
    outputs = llm.generate(inputs, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="test")
    parser.add_argument(
        "--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct"
    )
    args, _ = parser.parse_known_args()

    main(args)
