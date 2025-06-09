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

# 1. set env before import mindspore.
from vllm_mindspore.scripts import env_setup
env_setup()

# 2. update the log configuration ahead of other modifications.
import vllm_mindspore.logger

from vllm_mindspore.platforms.ascend import AscendPlatform

ascend_platform = AscendPlatform()

import vllm.config

vllm.config.current_platform = ascend_platform

import vllm.platforms

vllm.platforms.current_platform = ascend_platform

import vllm.utils

vllm.utils.current_platform = ascend_platform

import vllm.attention.selector
vllm.attention.selector.current_platform = ascend_platform

import vllm.engine.arg_utils
from vllm_mindspore.engine.arg_utils import _is_v1_supported_oracle
vllm.engine.arg_utils.EngineArgs._is_v1_supported_oracle = _is_v1_supported_oracle

import vllm.v1.engine.core
from vllm_mindspore.v1.engine.core import shutdown
vllm.v1.engine.core.DPEngineCoreProc.shutdown = shutdown

from vllm_mindspore.utils import (
    make_tensor_with_pad,
    async_tensor_h2d,
    ascend_is_initialized,
    ms_memory_profiling,
)

vllm.utils.make_tensor_with_pad = make_tensor_with_pad
vllm.utils.async_tensor_h2d = async_tensor_h2d
vllm.utils.cuda_is_initialized = ascend_is_initialized
vllm.utils.memory_profiling = ms_memory_profiling

import vllm.executor

from vllm_mindspore.model_executor.models.registry import (
    MindSporeModelRegistry,
    _SUBPROCESS_COMMAND,
)

vllm.config.ModelRegistry = MindSporeModelRegistry

import vllm.model_executor

vllm.model_executor.models.ModelRegistry = MindSporeModelRegistry
vllm.model_executor.models.registry._SUBPROCESS_COMMAND = _SUBPROCESS_COMMAND

from vllm_mindspore.model_executor.model_loader.utils import get_ms_model_architecture

# To patching the get_model_architecture, should import it first.
from vllm.model_executor.model_loader import get_model_architecture

vllm.model_executor.model_loader.get_model_architecture = get_ms_model_architecture
vllm.model_executor.model_loader.utils.get_model_architecture = (
    get_ms_model_architecture
)
vllm.model_executor.model_loader.loader.get_model_architecture = (
    get_ms_model_architecture
)

from vllm_mindspore.model_executor.sampling_metadata import SamplingTensors

vllm.model_executor.sampling_metadata.async_tensor_h2d = async_tensor_h2d
vllm.model_executor.sampling_metadata.SamplingTensors.from_lists = SamplingTensors.from_lists
from vllm_mindspore.worker.cache_engine import (
    ms_allocate_kv_cache,
    ms_swap_in,
    ms_swap_out,
)

import vllm.worker.cache_engine

vllm.worker.cache_engine.CacheEngine._allocate_kv_cache = ms_allocate_kv_cache
vllm.worker.cache_engine.CacheEngine.swap_in = ms_swap_in
vllm.worker.cache_engine.CacheEngine.swap_out = ms_swap_out

from vllm_mindspore.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
)

vllm.model_executor.model_loader.loader.safetensors_weights_iterator = (
    safetensors_weights_iterator
)

from vllm_mindspore.worker.worker import _warm_up_model
from vllm_mindspore.worker.profile import (
    wrapper_worker_init,
    wrapper_worker_init_device,
)
from vllm.worker.worker import Worker

Worker._warm_up_model = _warm_up_model
Worker.__init__ = wrapper_worker_init(Worker.__init__)
Worker.init_device = wrapper_worker_init_device(Worker.init_device)

from vllm_mindspore.worker.model_runner import (
    _get_cuda_graph_pad_size,
    _dummy_run,
    _get_supported_attention_backends,
)

vllm.worker.model_runner.ModelInputForGPUBuilder._get_cuda_graph_pad_size = (
    _get_cuda_graph_pad_size
)
vllm.worker.model_runner.GPUModelRunnerBase._dummy_run = _dummy_run

import vllm.worker.multi_step_model_runner

vllm.worker.multi_step_model_runner._get_supported_attention_backends = (
    _get_supported_attention_backends
)

from vllm_mindspore.executor.multiproc_worker_utils import (
    get_mp_context as ms_get_mp_context,
    terminate_worker as ms_terminate_worker,
)

# To patching the get_mp_context, should import it first.
from vllm.executor.multiproc_worker_utils import get_mp_context

vllm.executor.multiproc_worker_utils.get_mp_context = ms_get_mp_context

import vllm.executor.multiproc_worker_utils

vllm.executor.multiproc_worker_utils.ProcessWorkerWrapper.terminate_worker = ms_terminate_worker

import vllm.v1.executor.multiproc_executor
vllm.v1.executor.multiproc_executor.get_mp_context = ms_get_mp_context
import vllm.v1.utils
vllm.v1.utils.get_mp_context = ms_get_mp_context

from vllm_mindspore.executor.ray_gpu_executor import (
    ms_init_workers_ray,
    initialize_ray_cluster,
)

from vllm.executor.ray_distributed_executor import RayDistributedExecutor

RayDistributedExecutor._init_workers_ray = ms_init_workers_ray

vllm.executor.ray_distributed_executor.initialize_ray_cluster = initialize_ray_cluster
vllm.executor.ray_utils.initialize_ray_cluster = initialize_ray_cluster

import vllm.engine.llm_engine
import vllm.engine.async_llm_engine

vllm.engine.llm_engine.initialize_ray_cluster = initialize_ray_cluster
vllm.engine.async_llm_engine.initialize_ray_cluster = initialize_ray_cluster


from .config import _verify_quantization, _verify_args, vllm_config_post_init, model_post_init, \
    _get_and_verify_dtype, stateless_init_dp_group, has_unfinished_dp

vllm.config.ModelConfig._verify_quantization = _verify_quantization
vllm.config.VllmConfig.__post_init__ = vllm_config_post_init
vllm.config.SchedulerConfig._verify_args = _verify_args
vllm.config.CompilationConfig.model_post_init = model_post_init
vllm.config._get_and_verify_dtype = _get_and_verify_dtype
vllm.config.ParallelConfig.stateless_init_dp_group = stateless_init_dp_group
vllm.config.ParallelConfig.has_unfinished_dp = has_unfinished_dp

from .utils import update_modules
from vllm_mindspore.attention.backends import ms_attn
update_modules("vllm.attention.backends.flash_attn", ms_attn)

from vllm_mindspore.worker.spec_decode_worker import (
    spec_decode_worker_init,
    _run_no_spec,
    _verify_tokens,
    _create_output,
    _merge_outputs,
)
from vllm.spec_decode.spec_decode_worker import SpecDecodeWorker
SpecDecodeWorker.__init__ = spec_decode_worker_init
SpecDecodeWorker._verify_tokens = _verify_tokens
SpecDecodeWorker._run_no_spec = _run_no_spec

from vllm.model_executor.layers.spec_decode_base_sampler import SpecDecodeBaseSampler
SpecDecodeBaseSampler._create_output = _create_output

from vllm.spec_decode.top1_proposer import Top1Proposer
Top1Proposer._merge_outputs = _merge_outputs

from vllm_mindspore.model_executor.layers.rejection_sampler import _smallest_positive_value, _multinomial
from vllm.model_executor.layers.rejection_sampler import RejectionSampler
RejectionSampler._smallest_positive_value = _smallest_positive_value
RejectionSampler._smallest_positive_value.__set_name__(RejectionSampler, '_smallest_positive_value')
vllm.model_executor.layers.rejection_sampler._multinomial = _multinomial

######### for multi-model
from vllm_mindspore.inputs.registry import call_hf_processor
from vllm.inputs.registry import InputProcessingContext
InputProcessingContext.call_hf_processor = call_hf_processor

from vllm_mindspore.multimodal.inputs import as_kwargs
from vllm.multimodal.inputs import MultiModalKwargs
MultiModalKwargs.as_kwargs = as_kwargs

from vllm_mindspore.model_executor.layers.rotary_embedding import InferMRotaryEmbedding
vllm.model_executor.layers.rotary_embedding.MRotaryEmbedding = InferMRotaryEmbedding

from vllm_mindspore.v1.sample import rejection_sampler
update_modules("vllm.v1.sample.rejection_sampler", rejection_sampler)

from vllm_mindspore.v1.spec_decode import eagle
update_modules("vllm.v1.spec_decode.eagle", eagle)

from vllm_mindspore.v1.attention.backends import flash_attn
import vllm.v1.attention.backends
sys.modules['vllm.v1.attention.backends.flash_attn'] = flash_attn
import vllm.v1.attention.backends.flash_attn

import vllm.v1.worker.gpu_model_runner

from vllm_mindspore.v1.worker.gpu_model_runner import _prepare_inputs
vllm.v1.worker.gpu_model_runner.GPUModelRunner._prepare_inputs = _prepare_inputs

from vllm_mindspore.v1.worker.gpu_model_runner import _update_states
vllm.v1.worker.gpu_model_runner.GPUModelRunner._update_states = _update_states

from vllm_mindspore.v1.worker.gpu_model_runner import initialize_kv_cache
vllm.v1.worker.gpu_model_runner.GPUModelRunner.initialize_kv_cache = initialize_kv_cache

from vllm_mindspore.v1.worker.gpu_model_runner import wrapper_gpu_model_runner_execute_model
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
vllm.v1.worker.gpu_model_runner.GPUModelRunner.execute_model = \
    wrapper_gpu_model_runner_execute_model(GPUModelRunner.execute_model)

import vllm.v1.worker.block_table
from vllm_mindspore.v1.worker.block_table import BlockTable
vllm.v1.worker.block_table.BlockTable = BlockTable
vllm.v1.worker.gpu_input_batch.BlockTable = BlockTable

import vllm.v1.worker.gpu_input_batch
from vllm_mindspore.v1.worker.gpu_input_batch import _make_sampling_metadata, _make_prompt_token_ids_tensor
vllm.v1.worker.gpu_input_batch.InputBatch._make_sampling_metadata = _make_sampling_metadata
vllm.v1.worker.gpu_model_runner.InputBatch._make_sampling_metadata = _make_sampling_metadata
vllm.v1.worker.gpu_input_batch.InputBatch._make_prompt_token_ids_tensor = _make_prompt_token_ids_tensor
vllm.v1.worker.gpu_model_runner.InputBatch._make_prompt_token_ids_tensor = _make_prompt_token_ids_tensor

from vllm.v1.worker.gpu_worker import Worker
from vllm_mindspore.v1.worker.gpu_worker import init_device

Worker.__init__ = wrapper_worker_init(Worker.__init__)
Worker.init_device = wrapper_worker_init_device(init_device)


import vllm.v1.utils
from vllm_mindspore.v1.utils import copy_slice
vllm.v1.utils.copy_slice = copy_slice
vllm.v1.worker.gpu_input_batch.copy_slice = copy_slice

from vllm_mindspore.v1.sample.ops.penalties import _convert_to_tensors
import vllm.v1.sample.ops.penalties
vllm.v1.sample.ops.penalties._convert_to_tensors = _convert_to_tensors
import vllm.model_executor.layers.utils
from vllm_mindspore.model_executor.layers.utils import apply_penalties
vllm.model_executor.layers.utils.apply_penalties = apply_penalties
vllm.v1.sample.ops.penalties.apply_penalties = apply_penalties


from vllm_mindspore.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample, \
    apply_top_k_only, topk_topp_sampler_forward_native

import vllm.v1.sample.ops.topk_topp_sampler
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
TopKTopPSampler.forward_native = topk_topp_sampler_forward_native
vllm.v1.sample.ops.topk_topp_sampler.apply_top_k_top_p = apply_top_k_top_p
vllm.v1.sample.ops.topk_topp_sampler.random_sample = random_sample
vllm.v1.sample.ops.topk_topp_sampler.apply_top_k_only = apply_top_k_only
from vllm_mindspore.v1.sample.sampler import apply_temperature
import vllm.v1.sample.sampler
vllm.v1.sample.sampler.Sampler.apply_temperature = apply_temperature

from vllm_mindspore.distributed.shm_broadcast import initialize_ShmRingBuffer
from vllm.distributed.device_communicators.shm_broadcast import ShmRingBuffer
ShmRingBuffer.__init__ = initialize_ShmRingBuffer

from vllm_mindspore.v1.worker.gpu_worker import compile_or_warm_up_model
from vllm.v1.worker.gpu_worker import Worker
Worker.compile_or_warm_up_model = compile_or_warm_up_model

from vllm_mindspore.v1.core.sched.scheduler import update_from_output
from vllm.v1.core.sched.scheduler import Scheduler
Scheduler.update_from_output = update_from_output

from .utils import check_ready

from vllm_mindspore.engine.multiprocessing.engine import cleanup
import vllm.engine.multiprocessing.engine
vllm.engine.multiprocessing.engine.MQLLMEngine.cleanup = cleanup

check_ready()
