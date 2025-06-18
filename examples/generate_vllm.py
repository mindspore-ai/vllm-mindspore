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
"""Example for generate by vllm mindspore."""

import vllm_mindspore  # noqa: F401, E402

from vllm import LLM, SamplingParams  # noqa: E402


def main(args):
    # Sample prompts.
    prompts = ["I am", "Today is", "What is"]

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0.0, top_p=0.95, max_tokens=args.max_tokens
    )

    # Create an LLM.
    _kwargs = {"model_impl": "transformers"} if args.use_transformers else {}
    _kwargs.update(
        {"gpu_memory_utilization": args.gpu_memory_utilization}
        if args.gpu_memory_utilization
        else {}
    )
    _kwargs.update(
        {"max_model_len": args.max_model_len} if args.max_model_len else {}
    )
    llm = LLM(model=args.model_path, **_kwargs)
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    import ast
    import argparse

    parser = argparse.ArgumentParser(description="test")
    parser.add_argument(
        "--model_path", type=str, default="Qwen/Qwen2.5-32B-Instruct"
    )
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument(
        "--use_transformers", type=ast.literal_eval, default=False
    )
    args, _ = parser.parse_known_args()

    main(args)
