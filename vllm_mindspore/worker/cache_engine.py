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

import mindspore as ms
from mindspore import mutable
from vllm.logger import init_logger
from vllm_mindspore.utils import MsKVCache, get_valid_dtype


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
