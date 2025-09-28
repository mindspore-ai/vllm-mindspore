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
"""Main entry point for monkey patching vllm."""

# isort:skip_file

import sys
import warnings
import msadapter  # noqa: F401
from vllm_mindspore.ray_patch import patch_ray

patch_ray()

if "vllm" in sys.modules:
    # Check models variable in sub process, cannot raise here.
    warnings.warn(
        "vllm import before vllm_mindspore, vllm_mindspore cannot \
                   worker right!",
        stacklevel=2,
    )

# 1. set env before import mindspore.
from vllm_mindspore.scripts import env_setup

env_setup()

# 2. update the log configuration ahead of other modifications.
import vllm_mindspore.logger  # noqa F401

from vllm_mindspore.platforms.ascend import AscendPlatform

ascend_platform = AscendPlatform()

import vllm.config

vllm.config.current_platform = ascend_platform

import vllm.platforms

vllm.platforms.current_platform = ascend_platform

import vllm.utils

vllm.utils.current_platform = ascend_platform

import vllm.executor.ray_utils

vllm.executor.ray_utils.current_platform = ascend_platform

import vllm.attention.selector

vllm.attention.selector.current_platform = ascend_platform

import vllm.spec_decode.spec_decode_worker

vllm.spec_decode.spec_decode_worker.current_platform = ascend_platform

import vllm.spec_decode.metrics

vllm.spec_decode.metrics.current_platform = ascend_platform

import vllm.engine.arg_utils
from vllm_mindspore.engine.arg_utils import (_is_v1_supported_oracle,
                                             _set_default_args_v1)

vllm.engine.arg_utils.EngineArgs._is_v1_supported_oracle = (
    _is_v1_supported_oracle)
vllm.engine.arg_utils.EngineArgs._set_default_args_v1 = _set_default_args_v1

import vllm.v1.engine.core
from vllm_mindspore.v1.engine.core import shutdown

vllm.v1.engine.core.DPEngineCoreProc.shutdown = shutdown

from vllm_mindspore.v1.core.kv_cache_utils import get_kv_cache_config

vllm.v1.core.kv_cache_utils.get_kv_cache_config = get_kv_cache_config
vllm.v1.engine.core.get_kv_cache_config = get_kv_cache_config
from vllm_mindspore.v1.core.single_type_kv_cache_manager import (
    find_longest_cache_hit, spec_manager_map)

vllm.v1.core.single_type_kv_cache_manager.FullAttentionManager.\
    find_longest_cache_hit = find_longest_cache_hit
vllm.v1.core.single_type_kv_cache_manager.spec_manager_map = spec_manager_map

from vllm_mindspore.utils import (
    make_tensor_with_pad,
    async_tensor_h2d,
    ascend_is_initialized,
    ms_memory_profiling,
)

from vllm_mindspore.config import CacheDType, _CacheConfig, \
    get_current_and_parent_class_attr_docs

vllm.config.CacheConfig = _CacheConfig
vllm.config.CacheDType = CacheDType
vllm.config.get_attr_docs = get_current_and_parent_class_attr_docs
import vllm.engine.arg_utils

vllm.engine.arg_utils.CacheDType = CacheDType
vllm.engine.arg_utils.CacheConfig = _CacheConfig
vllm.engine.arg_utils.get_attr_docs = get_current_and_parent_class_attr_docs

vllm.utils.make_tensor_with_pad = make_tensor_with_pad
vllm.utils.async_tensor_h2d = async_tensor_h2d
vllm.utils.cuda_is_initialized = ascend_is_initialized
vllm.utils.memory_profiling = ms_memory_profiling

import vllm.lora.utils

from vllm_mindspore.model_executor.layers.linear import LinearBase
from vllm_mindspore.lora.utils import _all_lora_classes

vllm.lora.utils._all_lora_classes = _all_lora_classes
vllm.lora.utils.LinearBase = LinearBase

import vllm.lora.models
from vllm_mindspore.lora.models import (
    register_module,
    from_local_checkpoint,
    from_lora_tensors,
)

vllm.lora.models.LoRAModelManager.register_module = register_module
vllm.lora.models.LoRAModel.from_local_checkpoint = from_local_checkpoint
vllm.lora.models.LoRAModel.from_lora_tensors = from_lora_tensors

from vllm_mindspore.lora.layers import (
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    MergedQKVParallelLinearWithLoRA,
    QKVParallelLinearWithLoRA,
    RowParallelLinearWithLoRA,
)

import vllm.lora.layers

vllm.lora.layers.ColumnParallelLinearWithLoRA = ColumnParallelLinearWithLoRA
vllm.lora.layers.MergedColumnParallelLinearWithLoRA = (
    MergedColumnParallelLinearWithLoRA)
vllm.lora.layers.MergedQKVParallelLinearWithLoRA = (
    MergedQKVParallelLinearWithLoRA)
vllm.lora.layers.QKVParallelLinearWithLoRA = QKVParallelLinearWithLoRA
vllm.lora.layers.RowParallelLinearWithLoRA = RowParallelLinearWithLoRA

import vllm.executor

from vllm_mindspore.model_executor.models.registry import (
    MindSporeModelRegistry,
    _SUBPROCESS_COMMAND,
)

vllm.config.ModelRegistry = MindSporeModelRegistry

import vllm.model_executor

vllm.model_executor.models.ModelRegistry = MindSporeModelRegistry
vllm.model_executor.models.registry._SUBPROCESS_COMMAND = _SUBPROCESS_COMMAND

from vllm_mindspore.model_executor.model_loader.utils import (
    get_ms_model_architecture, )

# To patching the get_model_architecture, should import it first.
from vllm.model_executor.model_loader import get_model_architecture  # noqa F401

vllm.model_executor.model_loader.get_model_architecture = (
    get_ms_model_architecture)
vllm.model_executor.model_loader.utils.get_model_architecture = (
    get_ms_model_architecture)
vllm.model_executor.model_loader.default_loader.get_model_architecture = (
    get_ms_model_architecture)

from vllm_mindspore.model_executor.sampling_metadata import SamplingTensors

vllm.model_executor.sampling_metadata.async_tensor_h2d = async_tensor_h2d
vllm.model_executor.sampling_metadata.SamplingTensors.from_lists = (
    SamplingTensors.from_lists)
from vllm_mindspore.worker.cache_engine import (
    ms_allocate_kv_cache,
    ms_swap_in,
    ms_swap_out,
)

from vllm_mindspore.utils import get_dtype_size
import vllm.worker.cache_engine

vllm.worker.cache_engine.CacheEngine._allocate_kv_cache = ms_allocate_kv_cache
vllm.worker.cache_engine.CacheEngine.swap_in = ms_swap_in
vllm.worker.cache_engine.CacheEngine.swap_out = ms_swap_out
vllm.worker.cache_engine.get_dtype_size = get_dtype_size

import vllm.v1.kv_cache_interface

vllm.v1.kv_cache_interface.get_dtype_size = get_dtype_size

from vllm_mindspore.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator, )

vllm.model_executor.model_loader.default_loader.safetensors_weights_iterator = (
    safetensors_weights_iterator)

from vllm_mindspore.worker.worker import (_warm_up_model,
                                          wrapper_worker_bind_cpu)
from vllm_mindspore.worker.profile import (
    wrapper_worker_init,
    wrapper_worker_init_device,
)
from vllm.worker.worker import Worker as V0Worker

V0Worker._warm_up_model = _warm_up_model
V0Worker.__init__ = (wrapper_worker_bind_cpu(
    wrapper_worker_init(V0Worker.__init__)))
V0Worker.init_device = wrapper_worker_init_device(V0Worker.init_device)

from vllm_mindspore.worker.model_runner import (
    _get_cuda_graph_pad_size,
    _dummy_run,
    _get_supported_attention_backends,
)

vllm.worker.model_runner.ModelInputForGPUBuilder._get_cuda_graph_pad_size = (
    _get_cuda_graph_pad_size)
vllm.worker.model_runner.GPUModelRunnerBase._dummy_run = _dummy_run

import vllm.worker.multi_step_model_runner

vllm.worker.multi_step_model_runner._get_supported_attention_backends = (
    _get_supported_attention_backends)

from vllm_mindspore.executor.multiproc_worker_utils import (
    get_mp_context as ms_get_mp_context,
    terminate_worker as ms_terminate_worker,
)

# To patching the get_mp_context, should import it first.
from vllm.executor.multiproc_worker_utils import get_mp_context  # noqa F401

vllm.executor.multiproc_worker_utils.get_mp_context = ms_get_mp_context

import vllm.executor.multiproc_worker_utils

vllm.executor.multiproc_worker_utils.ProcessWorkerWrapper.terminate_worker = (
    ms_terminate_worker)

import vllm.v1.executor.multiproc_executor

vllm.v1.executor.multiproc_executor.get_mp_context = ms_get_mp_context
import vllm.v1.utils

vllm.v1.utils.get_mp_context = ms_get_mp_context

from vllm_mindspore.executor.ray_utils import (WORKER_SPECIFIC_ENV_VARS,
                                               MsRayWorkerWrapper,
                                               initialize_ray_cluster,
                                               core_engine_actor_manager_init)

from vllm.executor.ray_distributed_executor import RayDistributedExecutor

RayDistributedExecutor.WORKER_SPECIFIC_ENV_VARS = WORKER_SPECIFIC_ENV_VARS
vllm.executor.ray_distributed_executor.RayWorkerWrapper = MsRayWorkerWrapper
vllm.executor.ray_utils.initialize_ray_cluster = initialize_ray_cluster
vllm.executor.ray_distributed_executor.initialize_ray_cluster = (
    initialize_ray_cluster)
vllm.v1.utils.CoreEngineActorManager.__init__ = core_engine_actor_manager_init

from .config import (_verify_quantization, _verify_args, vllm_config_post_init,
                     vllm_config_get_quantization_config, model_post_init,
                     _get_and_verify_dtype, stateless_init_dp_group,
                     has_unfinished_dp, v1_process_validate_sampling_params)

vllm.config.ModelConfig._verify_quantization = _verify_quantization
vllm.config.VllmConfig.__post_init__ = vllm_config_post_init
vllm.config.VllmConfig._get_quantization_config = staticmethod(
    vllm_config_get_quantization_config)
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

from vllm_mindspore.sequence import (
    sequence_multi_modal_data,
    sequence_group_multi_modal_data,
)
from vllm.sequence import Sequence, SequenceGroup

Sequence.multi_modal_data = sequence_multi_modal_data
SequenceGroup.multi_modal_data = sequence_group_multi_modal_data

from vllm.model_executor.layers.spec_decode_base_sampler import (
    SpecDecodeBaseSampler, )

SpecDecodeBaseSampler._create_output = _create_output

from vllm.spec_decode.top1_proposer import Top1Proposer

Top1Proposer._merge_outputs = _merge_outputs

from vllm_mindspore.model_executor.layers.rejection_sampler import (
    _smallest_positive_value, )
from vllm.model_executor.layers.rejection_sampler import RejectionSampler

RejectionSampler._smallest_positive_value = _smallest_positive_value
RejectionSampler._smallest_positive_value.__set_name__(
    RejectionSampler, "_smallest_positive_value")

######### for multi-model
from vllm_mindspore.inputs.registry import call_hf_processor
from vllm.inputs.registry import InputProcessingContext

InputProcessingContext.call_hf_processor = call_hf_processor

from vllm_mindspore.multimodal.inputs import (as_kwargs, batched_reduce_data,
                                              flat_build_elems,
                                              flat_reduce_data, from_items,
                                              MultiModalFieldElem, _try_stack)

from vllm.multimodal.inputs import MultiModalBatchedField
from vllm.multimodal.inputs import MultiModalFlatField
from vllm.multimodal.inputs import MultiModalKwargs

MultiModalBatchedField._reduce_data = batched_reduce_data
MultiModalFlatField.build_elems = flat_build_elems
MultiModalFlatField._reduce_data = flat_reduce_data
MultiModalKwargs.as_kwargs = as_kwargs
MultiModalKwargs.from_items = from_items
MultiModalKwargs._try_stack = _try_stack

vllm.multimodal.inputs.MultiModalFieldElem = MultiModalFieldElem
vllm.v1.serial_utils.MultiModalFieldElem = MultiModalFieldElem

from vllm_mindspore.multimodal.base import from_seq_group
from vllm.multimodal.base import MultiModalPlaceholderMap

MultiModalPlaceholderMap.from_seq_group = from_seq_group

from vllm_mindspore.v1.serial_utils import _encode_tensor, _decode_tensor
from vllm.v1.serial_utils import MsgpackEncoder, MsgpackDecoder

MsgpackEncoder._encode_tensor = _encode_tensor
MsgpackDecoder._decode_tensor = _decode_tensor

from vllm_mindspore.model_executor.layers.rotary_embedding import (
    InferMRotaryEmbedding, )

vllm.model_executor.layers.rotary_embedding.MRotaryEmbedding = (
    InferMRotaryEmbedding)

# patch for V1
from vllm_mindspore.v1.attention.backends import ms_attn

update_modules("vllm.v1.attention.backends.flash_attn", ms_attn)

import vllm.v1.worker.gpu_model_runner

from vllm_mindspore.v1.worker.gpu_model_runner import _prepare_inputs

vllm.v1.worker.gpu_model_runner.GPUModelRunner._prepare_inputs = _prepare_inputs

from vllm_mindspore.v1.worker.gpu_model_runner import _calc_mrope_positions

vllm.v1.worker.gpu_model_runner.GPUModelRunner._calc_mrope_positions = \
    _calc_mrope_positions

from vllm_mindspore.v1.worker.gpu_model_runner import _update_states

vllm.v1.worker.gpu_model_runner.GPUModelRunner._update_states = _update_states

from vllm_mindspore.v1.worker.gpu_model_runner import (
    _allocate_kv_cache_tensors, get_kv_cache_spec, initialize_kv_cache_tensors)

vllm.v1.worker.gpu_model_runner.GPUModelRunner._allocate_kv_cache_tensors = (
    _allocate_kv_cache_tensors)
vllm.v1.worker.gpu_model_runner.GPUModelRunner.get_kv_cache_spec = (
    get_kv_cache_spec)
vllm.v1.worker.gpu_model_runner.GPUModelRunner.initialize_kv_cache_tensors = (
    initialize_kv_cache_tensors)
from vllm_mindspore.v1.worker.gpu_model_runner import _reshape_kv_cache_tensors

vllm.v1.worker.gpu_model_runner.GPUModelRunner._reshape_kv_cache_tensors = (
    _reshape_kv_cache_tensors)

from vllm_mindspore.v1.worker.gpu_model_runner import (
    wrapper_gpu_model_runner_execute_model, )
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

vllm.v1.worker.gpu_model_runner.GPUModelRunner.execute_model = (
    wrapper_gpu_model_runner_execute_model(GPUModelRunner.execute_model))

from vllm_mindspore.v1.worker.gpu_model_runner import get_dp_padding

vllm.v1.worker.gpu_model_runner.GPUModelRunner.get_dp_padding = get_dp_padding

from vllm_mindspore.forward_context import set_forward_context

vllm.v1.worker.gpu_model_runner.GPUModelRunner.set_forward_context = (
    set_forward_context)

import vllm.v1.worker.block_table
from vllm_mindspore.v1.worker.block_table import BlockTable

vllm.v1.worker.block_table.BlockTable = BlockTable
vllm.v1.worker.gpu_input_batch.BlockTable = BlockTable

import vllm.v1.worker.gpu_input_batch
from vllm_mindspore.v1.worker.gpu_input_batch import (
    _make_sampling_metadata,
    _make_prompt_token_ids_tensor,
)

vllm.v1.worker.gpu_input_batch.InputBatch._make_sampling_metadata = (
    _make_sampling_metadata)
vllm.v1.worker.gpu_model_runner.InputBatch._make_sampling_metadata = (
    _make_sampling_metadata)
vllm.v1.worker.gpu_input_batch.InputBatch._make_prompt_token_ids_tensor = (
    _make_prompt_token_ids_tensor)
vllm.v1.worker.gpu_model_runner.InputBatch._make_prompt_token_ids_tensor = (
    _make_prompt_token_ids_tensor)

import vllm.v1.utils
from vllm_mindspore.v1.utils import copy_slice, create_dp_placement_groups

vllm.v1.utils.copy_slice = copy_slice
vllm.v1.worker.gpu_input_batch.copy_slice = copy_slice
vllm.v1.utils.CoreEngineActorManager.create_dp_placement_groups = staticmethod(
    create_dp_placement_groups)

from vllm_mindspore.model_executor.layers.sampler import (
    _apply_top_k_top_p,
    _apply_min_p,
    _random_sample,
)
import vllm.model_executor.layers

vllm.model_executor.layers.sampler._apply_top_k_top_p = _apply_top_k_top_p
vllm.model_executor.layers.sampler._apply_min_p = _apply_min_p
vllm.model_executor.layers.sampler._random_sample = _random_sample

from vllm_mindspore.v1.sample.ops.penalties import _convert_to_tensors
import vllm.v1.sample.ops.penalties

vllm.v1.sample.ops.penalties._convert_to_tensors = _convert_to_tensors

from vllm_mindspore.model_executor.layers.utils import apply_penalties
import vllm.model_executor.layers.utils

vllm.model_executor.layers.sampler.apply_penalties = apply_penalties
vllm.model_executor.layers.utils.apply_penalties = apply_penalties
vllm.v1.sample.ops.penalties.apply_penalties = apply_penalties

from vllm_mindspore.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p,
    random_sample,
    apply_top_k_only,
    topk_topp_sampler_forward_native,
)

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

import vllm.distributed.device_communicators.base_device_communicator

vllm.distributed.device_communicators.base_device_communicator.\
    DeviceCommunicatorBase.prepare_communication_buffer_for_model = \
        lambda self, model : None

from vllm_mindspore.v1.worker.gpu_worker import compile_or_warm_up_model

from vllm.v1.worker.gpu_worker import Worker as V1Worker

V1Worker.__init__ = (wrapper_worker_bind_cpu(
    wrapper_worker_init(V1Worker.__init__)))
V1Worker.init_device = wrapper_worker_init_device(V1Worker.init_device)
V1Worker.compile_or_warm_up_model = compile_or_warm_up_model

from vllm_mindspore.v1.core.sched.scheduler import update_from_output
from vllm.v1.core.sched.scheduler import Scheduler

Scheduler.update_from_output = update_from_output

from vllm_mindspore.v1.executor.multiproc_executor import (
    executor_ensure_worker_termination, )
from vllm.v1.executor.multiproc_executor import MultiprocExecutor

MultiprocExecutor._ensure_worker_termination = staticmethod(
    executor_ensure_worker_termination)

from .utils import check_ready

from vllm_mindspore.engine.multiprocessing.engine import cleanup
import vllm.engine.multiprocessing.engine

vllm.engine.multiprocessing.engine.MQLLMEngine.cleanup = cleanup

from vllm_mindspore.entrypoints.openai.tool_parsers import (
    deepseekv3_tool_parser, )

sys.modules["vllm.entrypoints.openai.tool_parsers.deepseekv3_tool_parser"] = (
    deepseekv3_tool_parser)

from vllm_mindspore.entrypoints.__main__ import (
    patch_server_run_api_server_worker_proc, )

patch_server_run_api_server_worker_proc()

from vllm_mindspore.model_executor.models.registry import _normalize_archs
from vllm.model_executor.models.registry import _ModelRegistry

_ModelRegistry._normalize_archs = _normalize_archs

from vllm_mindspore.v1.engine.core_client import (MsCoreEngine,
                                                  get_core_engine_for_request,
                                                  add_request_async,
                                                  process_engine_outputs)

vllm.entrypoints.cli.serve.CoreEngine = MsCoreEngine
vllm.v1.engine.core_client.CoreEngine = MsCoreEngine
vllm.v1.utils.CoreEngine = MsCoreEngine

from vllm.v1.engine.core_client import DPAsyncMPClient

DPAsyncMPClient.get_core_engine_for_request = get_core_engine_for_request
DPAsyncMPClient.add_request_async = add_request_async
DPAsyncMPClient.process_engine_outputs = staticmethod(process_engine_outputs)

from vllm.v1.engine.processor import Processor

Processor._validate_sampling_params = v1_process_validate_sampling_params

check_ready()
