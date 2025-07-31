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

import gc

import torch
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.utils import MemorySnapshot

logger = init_logger(__name__)


def init_device(self):
    from vllm.config import get_current_vllm_config
    from vllm.model_executor import set_random_seed
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import (_check_if_gpu_supports_dtype,
                                           init_worker_distributed_environment)

    config = get_current_vllm_config()
    if config is not None and config.parallel_config.data_parallel_size > 1:
        # DLLM
        self.local_rank = (self.parallel_config.data_parallel_rank_local *
                           self.parallel_config.world_size + self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")
    else:
        self.device = torch.device(f"cuda:{self.local_rank}")
    logger.debug("self.local_rank: %s, self.device: %s", self.local_rank,
                 self.device)
    torch.cuda.set_device(self.device)

    _check_if_gpu_supports_dtype(self.model_config.dtype)
    gc.collect()
    torch.cuda.empty_cache()
    self.init_gpu_memory = torch.cuda.mem_get_info()[0]

    # Initialize the distributed environment.
    init_worker_distributed_environment(config, self.rank,
                                        self.distributed_init_method,
                                        self.local_rank)

    # Set random seed.
    set_random_seed(self.model_config.seed)

    # Construct the model runner
    self.model_runner = GPUModelRunner(self.vllm_config, self.device)
    self.init_snapshot = MemorySnapshot()
    self.requested_memory = (self.init_snapshot.total_memory *
                             self.cache_config.gpu_memory_utilization)


def compile_or_warm_up_model(self) -> None:
    # MindSpore does not support cuda graph. No need to warm up the model.
    # Since prefill is done previously, we do decode here.
    default_max_num_reqs = 1  # For MindSpore, we only do one more decode here.
    # Only pp_last_rank requires _dummy_sampler_run, and only pp_last_rank can _dummy_sampler_run.
    if get_pp_group().is_last_rank:
        self.model_runner._dummy_sampler_run(
            self.model_runner._dummy_run(num_tokens=default_max_num_reqs))
    else:
        self.model_runner._dummy_run(num_tokens=default_max_num_reqs)
