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
"""test mf qwen chunk prefill."""
import pytest
import os
from . import set_env

env_manager = set_env.EnvVarManager()
# def env
env_vars = {
    "MINDFORMERS_MODEL_CONFIG": "./config/predict_qwen2_5_7b_instruct.yaml",
    "ASCEND_CUSTOM_PATH": os.path.expandvars("$ASCEND_HOME_PATH/../"),
    "vLLM_MODEL_BACKEND": "MindFormers",
    "MS_ENABLE_LCCL": "off",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ASCEND_RT_VISIBLE_DEVICES": "0,1",
    "MS_ALLOC_CONF": "enable_vmm:True",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
    "VLLM_USE_V1": "0",
}
# set env
env_manager.setup_ai_environment(env_vars)
import vllm_mindspore
from vllm import LLM, SamplingParams


class TestMfQwen_chunk_prefill:
    """
    Test qwen.
    """

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.skip(reason="mindspore not ready")
    def test_mf_qwen_7b_chunk_prefill(self):
        """
        test case qwen_7b_chunk_prefill
        """

        # Sample prompts.
        batch_datas = [{
            "prompt": "I love Beijing, because it is a city with a long history and profound cultural heritage. Walking through "
                      "its ancient hutongs, one can almost feel the whispers of the past. The Forbidden City, an architectural "
                      "marvel that once housed emperors, stands as a testament to the city's imperial past. Meanwhile, the Great "
                      "Wall, though not within the city limits, is easily accessible from Beijing and offers a glimpse into the "
                      "strategic genius and resilience of ancient China.",
            "answer": " The city's blend of traditional and modern architecture, bustling markets, and vibrant street life make it "
                      "a unique and fascinating destination. In short, Beijing is a city"},
            {"prompt": "I love Beijing, because",
             "answer": " it is a city with a long history. Which of the following options correctly expresses this sentence?\nA. I love Beijing, because it is a city with a"},
        ]

        # Create a sampling params object.
        sampling_params = SamplingParams(temperature=0.0, max_tokens=32, top_k=1)

        # Create an LLM.
        llm = LLM(model="/home/workspace/mindspore_dataset/weight/Qwen2.5-7B-Instruct",
                  max_model_len=8192, max_num_seqs=16, max_num_batched_tokens=32,
                  block_size=32, gpu_memory_utilization=0.9, tensor_parallel_size=2,
                  enable_chunked_prefill=True)
        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        for batch_data in batch_datas:
            prompt = batch_data["prompt"]
            answer = batch_data["answer"]
            outputs = llm.generate(prompt, sampling_params)
            # Print the outputs.
            for i, output in enumerate(outputs):
                generated_text = output.outputs[0].text
                print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")
                assert generated_text == answer

        # unset env
        env_manager.unset_all()
