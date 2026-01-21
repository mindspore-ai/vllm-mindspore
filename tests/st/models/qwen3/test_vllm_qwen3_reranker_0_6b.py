# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 Huawei Technologites Co., Ltd
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

# isort:skip_file
"""test vllm qwen3 reranker 0.6b."""
import pytest
from unittest.mock import patch

import os

from tests.utils.common_utils import (teardown_function, setup_function,
                                      MODEL_PATH)
# def env
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
def test_vllm_qwen3_reranker_0_6b():
    """
    Test Summary:
        Test qwen3 reranker 0.6B with graph mode.
    Expected Result:
        Running successfully, the request result meets expectations
    Model Info:
        Qwen3-Reranker-0.6B
    """
    import vllm_mindspore
    import mindspore as ms

    from vllm import LLM

    prefix = '<|im_start|>system\nJudge whether the Document meets the \
        requirements based on the Query and the Instruct provided. \
            Note that the answer can only be "yes" or "no".\
                <|im_end|>\n<|im_start|>user\n'

    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
    document_template = "<Document>: {doc}{suffix}"

    instruction = (
        "Given a web search query, retrieve relevant passages that answer "
        "the query")

    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]

    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. "
        "It gives weight to physical objects and is responsible for "
        "the movement of planets around the sun.",
    ]

    queries = [
        query_template.format(prefix=prefix,
                              instruction=instruction,
                              query=query) for query in queries
    ]
    documents = [
        document_template.format(doc=doc, suffix=suffix) for doc in documents
    ]

    llm = LLM(model=MODEL_PATH["Qwen3-Reranker-0.6B"],
              runner="pooling",
              hf_overrides={
                  "architectures": ["Qwen3ForSequenceClassification"],
                  "classifier_from_token": ["no", "yes"],
                  "is_original_qwen3_reranker": True,
              },
              gpu_memory_utilization=0.9,
              max_model_len=4096)
    outputs = llm.score(queries, documents)

    result = ms.tensor([output.outputs.score for output in outputs])
    golden_result = ms.tensor([0.999534010887146, 0.9993603825569153])
    assert ms.mint.allclose(result, golden_result, rtol=1e-4)