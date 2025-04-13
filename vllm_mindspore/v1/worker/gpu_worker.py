# SPDX-License-Identifier: Apache-2.0
"""A GPU worker class"""

import gc
import torch
# import mindspore
from vllm.logger import init_logger

logger = init_logger(__name__)

def init_device(self):
    from vllm.config import get_current_vllm_config
    from vllm.model_executor import set_random_seed
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import (
        _check_if_gpu_supports_dtype, init_worker_distributed_environment)

    config = get_current_vllm_config()
    if config is not None and config.parallel_config.data_parallel_size > 1:
        device_id = self.parallel_config.data_parallel_rank_local * self.parallel_config.world_size + self.local_rank
        self.device = torch.device(f"cuda:{device_id}")
    else:
        self.device = torch.device(f"cuda:{self.local_rank}")
    torch.cuda.set_device(self.device)

    _check_if_gpu_supports_dtype(self.model_config.dtype)
    gc.collect()
    torch.cuda.empty_cache()
    self.init_gpu_memory = torch.cuda.mem_get_info()[0]

    # Initialize the distributed environment.
    init_worker_distributed_environment(self.parallel_config, self.rank,
                                        self.distributed_init_method,
                                        self.local_rank)

    # Set random seed.
    set_random_seed(self.model_config.seed)

    # Construct the model runner
    self.model_runner: GPUModelRunner = GPUModelRunner(
        self.vllm_config, self.device)
