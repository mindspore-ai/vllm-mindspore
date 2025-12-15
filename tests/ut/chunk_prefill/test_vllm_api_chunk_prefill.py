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
import pytest
import vllm_mindspore
from vllm import LLM, SamplingParams
from tests.utils.common_utils import (teardown_function, setup_function,
                                      MODEL_PATH)
from tests.utils.env_var_manager import EnvVarManager

env_manager = EnvVarManager()
env_manager.setup_mindformers_environment()

QWEN_7B_MODEL = MODEL_PATH["Qwen2.5-7B-Instruct"]


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_vllm_api_chunked_prefill_001():
    """
    Test Summary:
        Perform interface testing with default mode (chunked_prefill enabled
        by default), and set max_num_batched_tokens < max_model_len.
    Expected Result:
        Successful execution
    Model Info:
        Qwen2.5-7B-Instruct
    """
    prompts = "I love Beijing."
    sampling_params = SamplingParams(temperature=0.0,
                                     top_p=0.95,
                                     max_tokens=120)
    llm = LLM(QWEN_7B_MODEL, max_num_batched_tokens=521, max_model_len=1024)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        assert prompt == prompts
        assert output.finished is True


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_vllm_api_chunked_prefill_002():
    """
    Test Summary:
        Perform interface testing explicitly enable chunked_prefill,
        and set max_num_batched_tokens < max_num_seqs.
    Expected Result:
        Raises ValueError containing the message "must be greater than
        or equal to max_num_seqs".
    Model Info:
        Qwen2.5-7B-Instruct
    """
    with pytest.raises(ValueError,
                       match="must be greater than or equal to max_num_seqs"):
        LLM(QWEN_7B_MODEL,
            max_num_seqs=256,
            max_num_batched_tokens=55,
            enable_chunked_prefill=True)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_vllm_api_chunked_prefill_003():
    """
    Test Summary:
        Perform interface testing with default mode (chunked_prefill enabled
        by default), and set the max_num_batched_token with a minus number.
    Expected Result:
        Raises ValueError containing the message "Engine core initialization
        failed".
    Model Info:
        Qwen2.5-7B-Instruct
    """
    with pytest.raises(RuntimeError,
                       match="Engine core initialization failed"):
        LLM(QWEN_7B_MODEL,
            max_num_batched_tokens=-1,
            enable_chunked_prefill=True)
