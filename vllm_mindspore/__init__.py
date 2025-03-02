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

import sys
import warnings

if "vllm" in sys.modules:
    # Check models variable in sub process, cannot raise here.
    warnings.warn(
        "vllm import before vllm_mindspore, vllm_mindspore cannot worker right!"
    )

from vllm_mindspore.scripts import env_setup

env_setup()

from vllm_mindspore.platforms.ascend import AscendPlatform

ascend_platform = AscendPlatform()

import vllm.config

vllm.config.current_platform = ascend_platform

import vllm.platforms

vllm.platforms.current_platform = ascend_platform

import vllm.utils

vllm.utils.current_platform = ascend_platform

from vllm_mindspore.utils import (
    direct_register_custom_op,
    memory_profiling,
    make_tensor_with_pad,
    async_tensor_h2d,
    get_dtype_size,
    ascend_device_count_stateless,
    ascend_is_initialized,
)

vllm.utils.direct_register_custom_op = direct_register_custom_op
vllm.utils.memory_profiling = memory_profiling
vllm.utils.make_tensor_with_pad = make_tensor_with_pad
vllm.utils.async_tensor_h2d = async_tensor_h2d
vllm.utils.get_dtype_size = get_dtype_size
vllm.utils.cuda_device_count_stateless = ascend_device_count_stateless
vllm.utils.cuda_is_initialized = ascend_is_initialized
vllm.config.cuda_device_count_stateless = ascend_device_count_stateless

import vllm.executor

vllm.executor.cuda_device_count_stateless = ascend_device_count_stateless

from vllm_mindspore.model_executor.models.registry import (
    MindSporeModelRegistry,
    _run_in_subprocess,
)

import vllm.model_executor

vllm.model_executor.models.ModelRegistry = MindSporeModelRegistry
vllm.config.ModelRegistry = MindSporeModelRegistry

from vllm_mindspore.model_executor.model_loader.utils import get_ms_model_architecture
from vllm.model_executor.model_loader import get_model_architecture

vllm.model_executor.model_loader.get_model_architecture = get_ms_model_architecture
vllm.model_executor.model_loader.utils.get_model_architecture = (
    get_ms_model_architecture
)
vllm.model_executor.model_loader.loader.get_model_architecture = (
    get_ms_model_architecture
)
vllm.model_executor.models.registry._run_in_subprocess = _run_in_subprocess

from vllm_mindspore.model_executor.sampling_metadata import (
    SequenceGroupToSample,
    SamplingMetadataCache,
    SamplingMetadata,
)

vllm.model_executor.SamplingMetadataCache = SamplingMetadataCache
vllm.model_executor.SamplingMetadata = SamplingMetadata
vllm.model_executor.sampling_metadata.SequenceGroupToSample = SequenceGroupToSample
vllm.model_executor.sampling_metadata.SamplingMetadataCache = SamplingMetadataCache
vllm.model_executor.sampling_metadata.SamplingMetadata = SamplingMetadata

from vllm_mindspore.attention.selector import get_ms_attn_backend

import vllm.attention

vllm.attention.get_attn_backend = get_ms_attn_backend

from vllm_mindspore.worker.cache_engine import (
    ms_allocate_kv_cache,
    ms_swap_in,
    ms_swap_out,
    cache_engine_init,
    get_cache_block_size,
)

import vllm.worker.cache_engine

vllm.worker.cache_engine.CacheEngine._allocate_kv_cache = ms_allocate_kv_cache
vllm.worker.cache_engine.CacheEngine.__init__ = cache_engine_init
vllm.worker.cache_engine.CacheEngine.swap_in = ms_swap_in
vllm.worker.cache_engine.CacheEngine.swap_out = ms_swap_out
vllm.worker.cache_engine.CacheEngine.get_cache_block_size = get_cache_block_size

from vllm_mindspore.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
)

vllm.model_executor.model_loader.loader.safetensors_weights_iterator = (
    safetensors_weights_iterator
)

from vllm_mindspore.worker.worker import (
    _warm_up_model,
    determine_num_available_blocks,
)
from vllm.worker.worker import Worker

Worker._warm_up_model = _warm_up_model
Worker.determine_num_available_blocks = determine_num_available_blocks

from vllm_mindspore.worker.model_runner import _get_cuda_graph_pad_size, profile_run

vllm.worker.model_runner.ModelInputForGPUBuilder._get_cuda_graph_pad_size = (
    _get_cuda_graph_pad_size
)
vllm.worker.model_runner.GPUModelRunnerBase.profile_run = profile_run

from vllm_mindspore.distributed.parallel_state import (
    all_reduce_for_GroupCoordinator,
    init_model_parallel_group,
)

vllm.distributed.parallel_state.GroupCoordinator.all_reduce = (
    all_reduce_for_GroupCoordinator
)
vllm.distributed.parallel_state.init_model_parallel_group = init_model_parallel_group

from vllm_mindspore.executor.multiproc_worker_utils import (
    get_mp_context as ms_get_mp_context,
)
from vllm.executor.multiproc_worker_utils import get_mp_context

vllm.executor.multiproc_worker_utils.get_mp_context = ms_get_mp_context

from vllm_mindspore.executor.ray_gpu_executor import (
    ms_init_workers_ray,
    initialize_ray_cluster,
)

from vllm.executor.ray_gpu_executor import RayGPUExecutor

RayGPUExecutor._init_workers_ray = ms_init_workers_ray

vllm.executor.ray_utils.initialize_ray_cluster = initialize_ray_cluster

import vllm.engine.llm_engine
import vllm.engine.async_llm_engine

vllm.engine.llm_engine.initialize_ray_cluster = initialize_ray_cluster
vllm.engine.async_llm_engine.initialize_ray_cluster = initialize_ray_cluster


from .config import get_head_size, _verify_quantization, get_num_kv_heads

vllm.config.ModelConfig.get_head_size = get_head_size
vllm.config.ModelConfig._verify_quantization = _verify_quantization
vllm.config.ModelConfig.get_num_kv_heads = get_num_kv_heads

from .utils import check_ready

check_ready()
