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
"""CacheEngine class for managing the KV cache."""

from typing import List

from vllm.logger import init_logger

from vllm_mindspore.utils import MsKVCache, get_valid_dtype, is_use_mla, get_dtype_size

import mindspore as ms
from mindspore import mutable

logger = init_logger(__name__)


def create_block(shape, dtype, name=None, device=None):
    from mindspore.ops.function.array_func import empty as empty_tensor

    blocks = empty_tensor(*shape, dtype=dtype, device=device)
    return blocks


def ms_allocate_kv_cache(
    self,
    num_blocks: int,
    device: str,
) -> List[MsKVCache]:
    """Allocates KV cache on the specified device."""
    kv_cache_shape = self.attn_backend.get_kv_cache_shape(
        num_blocks, self.block_size, self.num_kv_heads, self.head_size
    )
    kv_cache: List[MsKVCache] = []

    self.dtype = get_valid_dtype(self.dtype)

    # TODO(tronzhang): A shape with (2, ...) for a kv tensor cannot support in mindspore's tensor and block operation, so split it to two tensor.
    for _ in range(self.num_attention_layers):
        device_type = "CPU" if device == "cpu" else "Ascend"
        current_cache = []
        for i in range(kv_cache_shape[0]):
            cache_blocks = create_block(
                kv_cache_shape[1:], self.dtype, device=device_type
            )
            current_cache.append(mutable(cache_blocks))
        kv_cache.append(mutable(tuple(current_cache)))
    return mutable(kv_cache)


def ms_swap_in(self, src_to_dst: ms.Tensor) -> None:
    for i in range(self.num_attention_layers):
        self.attn_backend.swap_blocks(
            self.cpu_cache[i], self.gpu_cache[i], src_to_dst, False
        )


def ms_swap_out(self, src_to_dst: ms.Tensor) -> None:
    for i in range(self.num_attention_layers):
        self.attn_backend.swap_blocks(
            self.gpu_cache[i], self.cpu_cache[i], src_to_dst, True
        )


def cache_engine_init(
    self,
    cache_config,
    model_config,
    parallel_config,
    device_config,
) -> None:

    from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType
    from vllm.attention import get_attn_backend

    self.cache_config = cache_config
    self.model_config = model_config
    self.parallel_config = parallel_config
    self.device_config = device_config

    self.head_size = model_config.get_head_size()
    # Models like Jamba, have mixed typed layers, E.g Mamba
    self.num_attention_layers = model_config.get_num_layers_by_block_type(
        parallel_config, LayerBlockType.attention
    )
    self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

    self.block_size = cache_config.block_size
    self.num_gpu_blocks = cache_config.num_gpu_blocks
    if self.num_gpu_blocks:
        self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
    self.num_cpu_blocks = cache_config.num_cpu_blocks
    if self.num_cpu_blocks:
        self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

    if cache_config.cache_dtype == "auto":
        self.dtype = model_config.dtype
    else:
        self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

    # Get attention backend.
    self.attn_backend = get_attn_backend(
        self.head_size,
        model_config.dtype,
        cache_config.cache_dtype,
        self.block_size,
        model_config.is_attention_free,
        use_mla=is_use_mla(model_config),
    )

    # Initialize the cache.
    self.gpu_cache = self._allocate_kv_cache(
        self.num_gpu_blocks, self.device_config.device_type
    )
    self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")


def get_cache_block_size(
    cache_config: "CacheConfig",
    model_config: "ModelConfig",
    parallel_config: "ParallelConfig",
) -> int:
    from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType

    head_size = model_config.get_head_size()
    num_heads = model_config.get_num_kv_heads(parallel_config)
    num_attention_layers = model_config.get_num_layers_by_block_type(
        parallel_config, LayerBlockType.attention
    )

    key_cache_block = cache_config.block_size * num_heads * head_size
    value_cache_block = key_cache_block if not is_use_mla(model_config) else 0
    total = num_attention_layers * (key_cache_block + value_cache_block)
    if cache_config.cache_dtype == "auto":
        dtype = model_config.dtype
    else:
        dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
    dtype_size = get_dtype_size(dtype)
    return dtype_size * total
