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

from typing import List, Dict, Optional

import torch
import torch.nn as nn
import weakref
import numpy as np
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceGroupMetadata
from vllm.config import VllmConfig
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalRegistry)
from vllm.worker.model_runner_base import ModelRunnerBase
from vllm.utils import (PyObjectCache, is_pin_memory_available)
from vllm.model_executor.models.utils import set_cpu_offload_max_bytes
from vllm.attention import get_attn_backend
from vllm.model_executor import SamplingMetadataCache
from vllm.prompt_adapter.worker_manager import LRUCacheWorkerPromptAdapterManager
from vllm.attention.backends.utils import CommonAttentionState
from vllm_mindspore.utils import STR_DTYPE_TO_TENSOR_DTYPE, is_use_mla

from mindspore.common import dtype as mstype
from mindspore import mutable, Tensor

logger = init_logger(__name__)

LORA_WARMUP_RANK = 8


def _get_cuda_graph_pad_size(
    self, num_seqs: int, max_decode_seq_len: int, max_encoder_seq_len: int = 0
) -> int:
    # No need to use cuda graph for mindspore.
    return -1


def profile_run(self) -> None:
    # Enable top-k sampling to reflect the accurate memory usage.
    sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
    max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
    max_num_seqs = self.scheduler_config.max_num_seqs
    # This represents the maximum number of different requests
    # that will have unique loras, an therefore the max amount of memory
    # consumption create dummy lora request copies from the lora request
    # passed in, which contains a lora from the lora warmup path.
    dummy_lora_requests: List[LoRARequest] = []
    dummy_lora_requests_per_seq: List[LoRARequest] = []
    if self.lora_config:
        assert self.lora_manager is not None
        with self.lora_manager.dummy_lora_cache():
            for idx in range(self.lora_config.max_loras):
                lora_id = idx + 1
                dummy_lora_request = LoRARequest(
                    lora_name=f"warmup_{lora_id}",
                    lora_int_id=lora_id,
                    lora_path="/not/a/real/path",
                )
                self.lora_manager.add_dummy_lora(dummy_lora_request,
                                                 rank=LORA_WARMUP_RANK)
                dummy_lora_requests.append(dummy_lora_request)
            dummy_lora_requests_per_seq = [
                dummy_lora_requests[idx % len(dummy_lora_requests)]
                for idx in range(max_num_seqs)
            ]

    # Profile memory usage with max_num_sequences sequences and the total
    # number of tokens equal to max_num_batched_tokens.
    seqs: List[SequenceGroupMetadata] = []
    # Additional GPU memory may be needed for multi-modal encoding, which
    # needs to be accounted for when calculating the GPU blocks for
    # vLLM blocker manager.
    # To exercise the worst scenario for GPU memory consumption,
    # the number of seqs (batch_size) is chosen to maximize the number
    # of images processed.

    max_mm_tokens = self.mm_registry.get_max_multimodal_tokens(
        self.model_config)
    if max_mm_tokens > 0:
        max_num_seqs_orig = max_num_seqs
        max_num_seqs = min(max_num_seqs,
                           max_num_batched_tokens // max_mm_tokens)
        if max_num_seqs < 1:
            expr = (f"min({max_num_seqs_orig}, "
                    f"{max_num_batched_tokens} // {max_mm_tokens})")
            logger.warning(
                "Computed max_num_seqs (%s) to be less than 1. "
                "Setting it to the minimum value of 1.", expr)
            max_num_seqs = 1

    batch_size = 0
    for group_id in range(max_num_seqs):
        seq_len = (max_num_batched_tokens // max_num_seqs +
                   (group_id < max_num_batched_tokens % max_num_seqs))
        batch_size += seq_len

        dummy_data = self.input_registry \
            .dummy_data_for_profiling(self.model_config,
                                      seq_len,
                                      self.mm_registry)

        seq = SequenceGroupMetadata(
            request_id=str(group_id),
            is_prompt=True,
            seq_data={group_id: dummy_data.seq_data},
            sampling_params=sampling_params,
            block_tables=None,
            lora_request=dummy_lora_requests_per_seq[group_id]
            if dummy_lora_requests_per_seq else None,
            multi_modal_data=dummy_data.multi_modal_data,
            multi_modal_placeholders=dummy_data.multi_modal_placeholders,
        )
        seqs.append(seq)

    # Run the model with the dummy inputs.
    num_layers = self.model_config.get_num_layers(self.parallel_config)
    # use an empty tensor instead of `None`` to force Dynamo to pass
    # it by reference, rather by specializing on the value ``None``.
    # the `dtype` argument does not matter, and we use `float32` as
    # a placeholder (it has wide hardware support).
    # it is important to create tensors inside the loop, rather than
    # multiplying the list, to avoid Dynamo from treating them as
    # tensor aliasing.

    # TODO(tronzhang): MindSpore's tensor view is limit now, delete this whole funtion patching latter.
    kv_cache_dtype = self.model_config.dtype if self.cache_config.cache_dtype == "auto" \
        else self.cache_config.cache_dtype
    kv_cache_dtype = STR_DTYPE_TO_TENSOR_DTYPE[kv_cache_dtype]
    block_size = self.cache_config.block_size
    num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
    head_size = self.model_config.get_head_size()
    kv_shape = [0, block_size, num_kv_heads, head_size]
    kv_caches = mutable([
        mutable((
            mutable(torch.tensor([], dtype=kv_cache_dtype, device=self.device).reshape(kv_shape)),
            mutable(torch.tensor([], dtype=kv_cache_dtype, device=self.device).reshape(kv_shape)),
        ))
        for _ in range(num_layers)
    ])
    finished_requests_ids = [seq.request_id for seq in seqs]
    model_input = self.prepare_model_input(
        seqs, finished_requests_ids=finished_requests_ids)
    intermediate_tensors = None
    if not get_pp_group().is_first_rank:
        intermediate_tensors = self.model.make_empty_intermediate_tensors(
            batch_size=batch_size,
            dtype=self.model_config.dtype,
            device=self.device)

    self.execute_model(model_input, kv_caches, intermediate_tensors)
    torch.cuda.synchronize()
    return

def gpu_model_runner_base_init(
    self,
    vllm_config: VllmConfig,
    kv_cache_dtype: Optional[str] = "auto",
    is_driver_worker: bool = False,
    return_hidden_states: bool = False,
    input_registry: InputRegistry = INPUT_REGISTRY,
    mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
):

    ModelRunnerBase.__init__(self, vllm_config)
    model_config = self.model_config
    cache_config = self.cache_config

    self.is_driver_worker = is_driver_worker
    self.return_hidden_states = return_hidden_states

    self.device = self.device_config.device
    self.pin_memory = is_pin_memory_available()

    self.kv_cache_dtype = kv_cache_dtype
    self.sliding_window = model_config.get_sliding_window()
    self.block_size = cache_config.block_size
    self.max_seq_len_to_capture = self.model_config.max_seq_len_to_capture
    self.max_batchsize_to_capture = \
        self.vllm_config.compilation_config.max_capture_size

    self.has_inner_state = model_config.has_inner_state

    # When using CUDA graph, the input block tables must be padded to
    # max_seq_len_to_capture. However, creating the block table in
    # Python can be expensive. To optimize this, we cache the block table
    # in numpy and only copy the actual input content at every iteration.
    # The shape of the cached block table will be
    # (max batch size to capture, max seq len to capture / block size).
    self.graph_block_tables = np.zeros(
        (self.max_batchsize_to_capture, self.get_max_block_per_batch()),
        dtype=np.int32)

    # Attention-free but stateful models like Mamba need a placeholder attn
    # backend, as the attention metadata is needed to manage internal state.
    # However we must bypass attention selection altogether for some models
    # used for speculative decoding to avoid a divide-by-zero in
    # model_config.get_head_size()
    num_attn_heads = self.model_config.get_num_attention_heads(
        self.parallel_config)
    needs_attn_backend = (num_attn_heads != 0
                          or self.model_config.is_attention_free)

    self.attn_backend = get_attn_backend(
        self.model_config.get_head_size(),
        self.model_config.dtype,
        self.kv_cache_dtype,
        self.block_size,
        self.model_config.is_attention_free,
        use_mla=is_use_mla(model_config),
    ) if needs_attn_backend else None
    if self.attn_backend:
        self.attn_state = self.attn_backend.get_state_cls()(
            weakref.proxy(self))
    else:
        self.attn_state = CommonAttentionState(weakref.proxy(self))

    # Multi-modal data support
    self.input_registry = input_registry
    self.mm_registry = mm_registry
    self.multi_modal_input_mapper = mm_registry \
        .create_input_mapper(model_config)
    self.mm_registry.init_mm_limits_per_prompt(self.model_config)

    # Lazy initialization
    self.model: nn.Module  # Set after load_model
    # Set after load_model.
    self.lora_manager: Optional[LRUCacheWorkerLoRAManager] = None
    self.prompt_adapter_manager: LRUCacheWorkerPromptAdapterManager = None

    set_cpu_offload_max_bytes(
        int(self.cache_config.cpu_offload_gb * 1024**3))

    # Used to cache python objects
    self.inter_data_cache: Dict[int, PyObjectCache] = {}

    # Using the PythonizationCache in Pipeline-Parallel clobbers the
    # SequenceGroupToSample object. In Pipeline-Parallel, we have
    # more than 1 Scheduler, resulting in a potential back-to-back
    # prepare_model_inputs() call. This clobbers the cached
    # SequenceGroupToSample objects, as we reset the cache during
    # every prepare_model_inputs() call.
    self.sampling_metadata_cache: SamplingMetadataCache = \
          SamplingMetadataCache() \
            if self.parallel_config.pipeline_parallel_size == 1 else None

MULTI_STEP_ATTENTION_BACKENDS = [
    "MS_MLA", "FLASH_ATTN", "ROCM_FLASH", "FLASHINFER", "NO_ATTENTION" 
]
MULTI_STEP_CHUNKED_PREFILL_ATTENTION_BACKENDS = ["FLASH_ATTN"]

def _get_supported_attention_backends(chunked_prefill_enabled: bool) \
    -> List[str]:
    if chunked_prefill_enabled:
        return MULTI_STEP_CHUNKED_PREFILL_ATTENTION_BACKENDS
    else:
        return MULTI_STEP_ATTENTION_BACKENDS