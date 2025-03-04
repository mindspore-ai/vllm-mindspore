# Copyright 2024 The vLLM team.
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://wwww.apache.org/licenses/LICENSE-2.0
#
# Unless required by application law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""test demo for st."""
import pytest


class TestDemo:
    """
    Test Demo for ST.
    """
    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_aaa(self):
        """
        test case aaa
        """
        from vllm import LLM, SamplingParams

        # Sample prompts.
        prompts = [
            "I am",
            "Today is",
            "Llama is"
        ]

        # Create a sampling params object.
        sampling_params = SamplingParams(temperature=0.0, top_p=0.95)

        # Create an LLM.
        llm = LLM(model="/home/workspace/mindspore_dataset/weight/Llama-2-7b-hf")
        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        outputs = llm.generate(prompts, sampling_params)
        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        assert len(outputs) == 3
