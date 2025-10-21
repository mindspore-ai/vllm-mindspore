# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/v1/worker/gpu_worker.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2025 The vLLM team.
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
"""Worker functions for ascend."""

from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger

logger = init_logger(__name__)


def compile_or_warm_up_model(self) -> None:
    # MindSpore does not support cuda graph. No need to warm up the model.
    # Since prefill is done previously, we do decode here.
    default_max_num_reqs = 1  # For MindSpore, we only do one more decode here.

    if hasattr(self.model_runner.model, 'set_chunked_flags'):
        logger.info("Warmup for chunked graph.")
        self.model_runner._dummy_run(num_tokens=default_max_num_reqs)

    # Only pp_last_rank has lm_head, which is required by _dummy_sampler_run.
    if get_pp_group().is_last_rank:
        self.model_runner._dummy_sampler_run(
            self.model_runner._dummy_run(num_tokens=default_max_num_reqs))
    else:
        self.model_runner._dummy_run(num_tokens=default_max_num_reqs)
