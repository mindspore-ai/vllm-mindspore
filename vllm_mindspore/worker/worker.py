#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Worker functions"""
import gc
import os
import math
from typing import Tuple, Optional

import torch

from vllm.config import VllmConfig
from vllm.distributed import (
    ensure_kv_transfer_initialized,
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
)

from vllm.logger import init_logger

from vllm_mindspore.utils import is_mindformers_model_backend
from vllm.model_executor import set_random_seed
from vllm.sequence import SequenceGroupMetadata
from vllm.sampling_params import SamplingParams


logger = init_logger(__name__)


def _prepare_input_for_warmup(model_config, model_runner, cache_engine, is_prefill):
    bs = 1
    seq_len = model_config.max_seq_len_to_capture if is_prefill else 1
    dummy_data = model_runner.input_registry.dummy_data_for_profiling(model_config, seq_len, model_runner.mm_registry)
    block_tables = [i for i in range(math.ceil(seq_len / cache_engine.block_size))]
    seqs = [
        SequenceGroupMetadata(
            request_id=str(idx),
            is_prompt=is_prefill,
            seq_data={idx: dummy_data.seq_data},
            sampling_params=SamplingParams(),
            block_tables={idx: block_tables},
            lora_request=None,
            multi_modal_data=None,
            multi_modal_placeholders=None,
        )
        for idx in range(bs)
    ]

    model_input = model_runner.prepare_model_input(seqs)
    return model_input


def _warm_up_model(self) -> None:
    # cache_engine is a list with length equal to the size of pipeline-parallel, and only pp=1 is supported.
    kv_cache = self.cache_engine[0].gpu_cache

    # warmup for prefill
    model_input = _prepare_input_for_warmup(self.model_config, self.model_runner, self.cache_engine[0], True)
    self.model_runner.execute_model(model_input, kv_cache, None)
    torch.cuda.synchronize()
    # warmup for decode
    model_input = _prepare_input_for_warmup(self.model_config, self.model_runner, self.cache_engine[0], False)
    self.model_runner.execute_model(model_input, kv_cache, None)
    torch.cuda.synchronize()

    # Reset the seed to ensure that the random state is not affected by
    # the model initialization and profiling.
    set_random_seed(self.model_config.seed)


def determine_num_available_blocks(self) -> Tuple[int, int]:
    """Profiles the peak memory usage of the model to determine how many
    KV blocks may be allocated without OOMs.

    The engine will first conduct a profiling of the existing memory usage.
    Then, it calculate the maximum possible number of GPU and CPU blocks
    that can be allocated with the remaining free memory.

    .. tip::
        You may limit the usage of GPU memory
        by adjusting the `gpu_memory_utilization` parameter.
    """
    from vllm.utils import GiB_bytes, memory_profiling

    # Profile the memory usage of the model and get the maximum number of
    # cache blocks that can be allocated with the remaining free memory.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    total_gpu_memory = int(os.environ["ASCEND_TOTAL_MEMORY_GB"]) if os.getenv("ASCEND_TOTAL_MEMORY_GB") else 64
    total_gpu_memory = total_gpu_memory * 1024 * 1024 * 1024

    if os.getenv("vLLM_MODEL_MEMORY_USE_GB"):
        memory_use_for_model_run = int(os.environ["vLLM_MODEL_MEMORY_USE_GB"]) * 1024 * 1024 * 1024
    else:
        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        _, total_gpu_memory = torch.cuda.mem_get_info()
        with memory_profiling(
            baseline_memory_in_bytes=total_gpu_memory - self.init_gpu_memory,
            weights_memory_in_bytes=self.model_runner.model_memory_usage,
        ) as result:
            self.model_runner.profile_run()
            torch.cuda.synchronize()

        self._assert_memory_footprint_increased_during_profiling()

        memory_use_for_model_run = result.non_kv_cache_memory_in_bytes

    memory_for_current_instance = (
        total_gpu_memory * self.cache_config.gpu_memory_utilization
    )
    available_kv_cache_memory = memory_for_current_instance - memory_use_for_model_run

    # Calculate the number of blocks that can be allocated with the
    # profiled peak memory.
    cache_block_size = self.get_cache_block_size_bytes()
    if cache_block_size == 0:
        num_gpu_blocks = 0
        num_cpu_blocks = 0
    else:
        num_gpu_blocks = int(available_kv_cache_memory // cache_block_size)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes // cache_block_size)
    num_gpu_blocks = max(num_gpu_blocks, 0)
    num_cpu_blocks = max(num_cpu_blocks, 0)

    if os.getenv("vLLM_MODEL_MEMORY_USE_GB"):
        msg = (
            f"The current vLLM instance can use "
            "total_gpu_memory "
            f"({(total_gpu_memory / GiB_bytes):.2f}GiB)"
            " x gpu_memory_utilization "
            f"({self.cache_config.gpu_memory_utilization:.2f})"
            f" = {(memory_for_current_instance / GiB_bytes):.2f}GiB\n"
            "set model use memory "
            f"{(memory_use_for_model_run):.2f}GiB;"
            " the rest of the memory reserved for KV Cache is "
            f"{(available_kv_cache_memory / GiB_bytes):.2f}GiB."
        )
    else:
        msg = (
            f"Memory profiling takes {result.profile_time:.2f} seconds\n"
            "the current vLLM instance can use "
            "total_gpu_memory "
            f"({(total_gpu_memory / GiB_bytes):.2f}GiB)"
            " x gpu_memory_utilization "
            f"({self.cache_config.gpu_memory_utilization:.2f})"
            f" = {(memory_for_current_instance / GiB_bytes):.2f}GiB\n"
            "model weights take "
            f"{(result.weights_memory_in_bytes / GiB_bytes):.2f}GiB;"
            " non_torch_memory takes "
            f"{(result.non_torch_increase_in_bytes / GiB_bytes):.2f}GiB;"
            " PyTorch activation peak memory takes "
            f"{(result.torch_peak_increase_in_bytes / GiB_bytes):.2f}GiB;"
            " the rest of the memory reserved for KV Cache is "
            f"{(available_kv_cache_memory / GiB_bytes):.2f}GiB."
        )

    logger.info(msg)

    # Final cleanup
    if self.model_runner.lora_manager:
        self.model_runner.remove_all_loras()
    gc.collect()

    return num_gpu_blocks, num_cpu_blocks
