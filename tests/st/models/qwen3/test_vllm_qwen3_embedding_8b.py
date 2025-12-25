# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologites Co., Ltd
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
"""test vllm qwen3 embedding 8b."""
import pytest
from unittest.mock import patch

import os

from tests.utils.env_var_manager import EnvVarManager
from tests.utils.common_utils import MODEL_PATH

env_manager = EnvVarManager()
env_manager.setup_mindformers_environment()
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
def test_vllm_qwen3_embedding_8b():
    """
    Test Summary:
        Test qwen3 embedding 8B with graph mode.
    Expected Result:
        Running successfully, the request result meets expectations
    Model Info:
        Qwen3-Embedding-8B
    """
    import vllm_mindspore
    import mindspore as ms

    from vllm import LLM

    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'

    # Each query must come with a one-sentence instruction that
    # describes the task
    task = 'Given a web search query, retrieve relevant passages that ' \
    'answer the query'
    queries = [
        get_detailed_instruct(task, 'What is the capital of China?'),
        get_detailed_instruct(task, 'Explain gravity')
    ]
    # No need to add instruction for retrieval documents
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. "
        "It gives weight to physical objects and is responsible for "
        "the movement of planets around the sun."
    ]
    input_texts = queries + documents
    model = LLM(model=MODEL_PATH["Qwen3-Embedding-8B"],
                task="embed",
                gpu_memory_utilization=0.5)
    outputs = model.embed(input_texts)
    embeddings = ms.tensor([o.outputs.embedding for o in outputs])
    scores = (embeddings[:2] @ embeddings[2:].T)
    expected_scores = ms.tensor([[0.7482624650001526, 0.07556197047233582],
                                 [0.08875375241041183, 0.6300010681152344]])
    assert ms.mint.allclose(scores, expected_scores, atol=5e-3)
