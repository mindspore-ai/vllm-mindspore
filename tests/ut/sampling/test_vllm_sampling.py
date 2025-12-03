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
"""test vllm sampling."""
import pytest
from unittest.mock import patch

import os
from tests.utils.common_utils import teardown_function, setup_function

# def env
env_vars = {
    "MS_ENABLE_LCCL": "off",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
    "VLLM_MS_MODEL_BACKEND": "Native"
}


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
def test_vllm_sampling_n_logprobs():
    """
    parameter n and logprobs test case
    """
    import vllm_mindspore
    from vllm import LLM, SamplingParams

    # Sample prompts.
    prompts = ["介绍一下北京。"]

    # Create a sampling params object.
    sampling_params = SamplingParams(max_tokens=10, n=2, logprobs=2)

    # Create an LLM.
    llm = LLM(
        model="/home/workspace/mindspore_dataset/weight/Qwen2.5-7B-Instruct",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2)
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    expect_logprobs_nums = [2, 3]
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        for seq_idx, seq in enumerate(output.outputs):
            generated_text = seq.text
            print(f"candidate {seq_idx}: \nPrompt: {prompt!r}, "
                  f"Generated text: {generated_text!r}")
            assert all([
                len(sample_logprob) in expect_logprobs_nums
                for sample_logprob in seq.logprobs
            ])
        assert len(output.outputs) == 2
