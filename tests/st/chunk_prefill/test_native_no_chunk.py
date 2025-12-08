# SPDX-License-Identifier: Apache-2.0
#
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
"""Test Prefill-first Scheduler."""
import os
from typing import Optional
import pytest
from unittest.mock import patch
import vllm_mindspore
from vllm import LLM, SamplingParams
from tests.utils.common_utils import (teardown_function, setup_function,
                                      MODEL_PATH)

EOS_TOKEN_ID = 151645
QWEN3_MODEL_PATH = MODEL_PATH["Qwen3-0.6B"]
env_vars = {
    "VLLM_MS_MODEL_BACKEND": "Native",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
}


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_prefill_first_schedule_qwen_offline():
    """
    Test Summary:
        关闭chunked_prefill时和prefixcache, 测试按prefill优先调度策略
    Expected Result:
        模型推理精度正确;
    Model Info:
        Qwen3-0.6B
    """
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4, top_k=1)

    with pytest.raises(ValueError):
        # max_num_batched_tokens >= max_model_len when chunked_prefill=False
        llm = LLM(model=QWEN3_MODEL_PATH,
                  gpu_memory_utilization=0.9,
                  tensor_parallel_size=1,
                  max_num_seqs=10,
                  max_num_batched_tokens=128,
                  max_model_len=256,
                  enable_prefix_caching=False,
                  enable_chunked_prefill=False)

    # Create an LLM.
    llm = LLM(model=QWEN3_MODEL_PATH,
              gpu_memory_utilization=0.9,
              tensor_parallel_size=1,
              max_num_seqs=10,
              max_num_batched_tokens=256,
              max_model_len=256,
              enable_prefix_caching=False,
              enable_chunked_prefill=False)

    # norm requests will be scheduled.
    prompts1 = [
        "<|im_start|>user\n将文本分类为中性、负面或正面。 "
        "\n文本：我认为这次假期还可以。 \n情感：<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ] * 5
    outputs = llm.generate(prompts1, sampling_params)
    excepted_list = ['情感：中性']
    for _, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        assert generated_text == excepted_list[
            0], f"Expected: {excepted_list[0]}, but got: {generated_text}"
