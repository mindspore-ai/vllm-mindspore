#!/usr/bin/env python3
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
"""
This example shows how to use the multi-LoRA functionality
for offline inference.

"""
import pytest
import os
from tests.st.python import set_env

env_manager = set_env.EnvVarManager()
# def env
env_vars = {
    "ASCEND_CUSTOM_PATH": os.path.expandvars("$ASCEND_HOME_PATH/../"),
    "MS_ENABLE_LCCL": "off",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "MS_ALLOC_CONF": "enable_vmm:True",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
    "VLLM_USE_V1": "1",
}
# set env
env_manager.setup_ai_environment(env_vars)
import vllm_mindspore
from typing import List, Optional, Tuple

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest


def create_test_prompts(
        lora_path: str
) -> List[Tuple[str, SamplingParams, Optional[LoRARequest]]]:
    """Create a list of test prompts with their sampling parameters.
    """
    return [
        ("违章停车与违法停车是否有区别？",
         SamplingParams(temperature=0.0, top_p=1, top_k=-1,
                        max_tokens=10), LoRARequest("sql-lora1", 1,
                                                    lora_path)),
    ]


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams,
                     Optional[LoRARequest]]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, lora_request = test_prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               lora_request=lora_request)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                print(f'text is: {request_output.outputs[0].text}', flush=True)
                assert " 从法律上来说，违章停车和违法" in request_output.outputs[0].text


def initialize_engine() -> LLMEngine:
    """Initialize the LLMEngine."""
    # max_loras: controls the number of LoRAs that can be used in the same
    #   batch. Larger numbers will cause higher memory usage, as each LoRA
    #   slot requires its own preallocated tensor.
    # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
    #   numbers will cause higher memory usage. If you know that all LoRAs will
    #   use the same rank, it is recommended to set this as low as possible.
    # max_cpu_loras: controls the size of the CPU LoRA cache.
    engine_args = EngineArgs(
        model="/home/workspace/mindspore_dataset/weight/Qwen2.5-7B-Instruct",
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64,
        max_cpu_loras=2,
        max_num_seqs=256,
        max_model_len=256,
        max_num_batched_tokens=400)
    return LLMEngine.from_engine_args(engine_args)


def test_multilora_inference():
    """test function that sets up and runs the prompt processing."""
    engine = initialize_engine()
    lora_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-7B-Lora-Law"
    test_prompts = create_test_prompts(lora_path)
    process_requests(engine, test_prompts)
    env_manager.unset_all()
