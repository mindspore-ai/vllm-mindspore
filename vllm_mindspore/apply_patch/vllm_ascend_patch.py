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
"""Main entry point for monkey patching vllm-ascend."""

# isort:skip_file
import copy
import math
from typing import (Dict, List, Optional, Union, cast)

import torch
import torch_npu
import torch.distributed as dist
from vllm.config import (VllmConfig)
from vllm.distributed.parallel_state import (
    is_global_first_rank)
from vllm.logger import logger
from vllm.utils import (get_dtype_size)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec, KVCacheConfig,
    KVCacheGroupSpec, KVCacheSpec,
    MambaSpec)
from vllm.v1.worker.utils import bind_kv_cache


def _initialize_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
    """
    Initialize the memory buffer for KV cache.

    Args:
        kv_cache_config: The KV cache config
    Returns:
        Dict[str, torch.Tensor]: A map between layer names to their
        corresponding memory buffer for KV cache.
    """
    # init kv cache tensors
    kv_cache_raw_tensors: dict[str, Union[torch.Tensor,
                                            Optional[torch.Tensor]]] = {}
    # llmdatadist need the addr of cache tensor be aligned with 2M
    alignment = 2 * 1024 * 1024
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        # TODO: REFACTOR ME to sharing hybrid cache
        for idx in range(len(kv_cache_tensor.shared_by)):
            layer_name = kv_cache_tensor.shared_by[idx]
            if "linear_attn" in layer_name:
                # for mamba linear attention
                for layer_name_inner in kv_cache_tensor.shared_by:
                    if ("attn" in layer_name_inner and "linear_attn" not in layer_name_inner) or \
                        layer_name_inner in kv_cache_raw_tensors.keys():
                        continue
                    if self.vllm_config.kv_transfer_config is None:
                        tensor = torch.zeros(kv_cache_tensor.size,
                                                dtype=torch.int8,
                                                device=self.device)
                    else:
                        cache_size_aligned = kv_cache_tensor.size + alignment
                        tensor = torch.zeros(cache_size_aligned,
                                                dtype=torch.int8,
                                                device=self.device)
                        tensor = self._align_memory(
                            tensor, alignment)[:kv_cache_tensor.size]
                    kv_cache_raw_tensors[layer_name_inner] = tensor
            # mod for vllm-mindspore
            # elif "attn" in layer_name:
            else:
                # for other attentions, e.g., self_attn, sliding window attn
                if self.vllm_config.kv_transfer_config is None:
                    k_tensor = torch.zeros(kv_cache_tensor.size // 2,
                                            dtype=torch.int8,
                                            device=self.device)
                    v_tensor = torch.zeros(kv_cache_tensor.size // 2,
                                            dtype=torch.int8,
                                            device=self.device)
                else:
                    cache_size = kv_cache_tensor.size // 2
                    cache_size_aligned = kv_cache_tensor.size // 2 + alignment
                    k_tensor = torch.zeros(cache_size_aligned,
                                            dtype=torch.int8,
                                            device=self.device)
                    v_tensor = torch.zeros(cache_size_aligned,
                                            dtype=torch.int8,
                                            device=self.device)
                    k_tensor = self._align_memory(k_tensor,
                                                    alignment)[:cache_size]
                    v_tensor = self._align_memory(v_tensor,
                                                    alignment)[:cache_size]
                kv_cache_raw_tensors[layer_name] = (k_tensor, v_tensor)

    layer_names = set()
    for group in kv_cache_config.kv_cache_groups:
        for layer_name in group.layer_names:
            if layer_name in self.runner_only_attn_layers:
                continue
            layer_names.add(layer_name)
    assert layer_names == set(kv_cache_raw_tensors.keys(
    )), "Some layers are not correctly initialized"

    kv_caches: Dict[str, torch.Tensor] = {}
    for group in self._kv_cache_spec_attn_group_iterator():
        kv_cache_spec = group.kv_cache_spec
        attn_backend = group.backend
        for layer_name in group.layer_names:
            if layer_name in self.runner_only_attn_layers:
                continue

            # TODO: remove this after the OOM issue is located and fixed, otherwise, some model may
            # encounter OOM issue
            if isinstance(kv_cache_spec, FullAttentionSpec):
                raw_k_tensor, raw_v_tensor = kv_cache_raw_tensors[  # type: ignore
                    layer_name]
                assert raw_k_tensor is not None
                assert raw_v_tensor is not None
                assert (raw_k_tensor.numel() + raw_v_tensor.numel()
                        ) % kv_cache_spec.page_size_bytes == 0
                num_blocks = (raw_k_tensor.numel() + raw_v_tensor.numel()
                                ) // kv_cache_spec.page_size_bytes

                # `num_blocks` is the number of blocks the model runner can use.
                # `kv_cache_config.num_blocks` is the number of blocks that
                # KVCacheManager may allocate.
                # Since different GPUs may have different number of layers and
                # different memory capacities, `num_blocks` can be different on
                # different GPUs, and `kv_cache_config.num_blocks` is set to
                # the min of all `num_blocks`. Verify it here.
                assert num_blocks >= kv_cache_config.num_blocks

                if self.vllm_config.additional_config.get(
                        "kv_cache_dtype", None) == 'int8':
                    kv_cache_shape = attn_backend.get_bsh_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size)
                elif hasattr(attn_backend, "get_supported_block_size"
                                ) and self.use_hybrid_blocks:
                    block_size = attn_backend.get_supported_block_size()[0]

                    block_size_chunk = kv_cache_spec.block_size // block_size
                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        num_blocks * block_size_chunk, block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size)
                else:
                    kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size)
                dtype = kv_cache_spec.dtype
                k_cache = raw_k_tensor.view(dtype).view(kv_cache_shape[1:])
                k_cache = self._convert_torch_format(k_cache)
                v_cache = raw_v_tensor.view(dtype).view(kv_cache_shape[1:])
                v_cache = self._convert_torch_format(v_cache)
                kv_caches[layer_name] = (k_cache, v_cache)
            elif isinstance(kv_cache_spec, MambaSpec):
                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor is not None
                assert raw_tensor.numel(
                ) % kv_cache_spec.page_size_bytes == 0
                num_blocks = raw_tensor.numel(
                ) // kv_cache_spec.page_size_bytes

                # `num_blocks` is the number of blocks the model runner can use.
                # `kv_cache_config.num_blocks` is the number of blocks that
                # KVCacheManager may allocate.
                # Since different GPUs may have different number of layers and
                # different memory capacities, `num_blocks` can be different on
                # different GPUs, and `kv_cache_config.num_blocks` is set to
                # the min of all `num_blocks`. Verify it here.
                assert num_blocks >= kv_cache_config.num_blocks

                state_tensors = []
                storage_offset_bytes = 0
                for (shape, dtype) in zip(kv_cache_spec.shapes,
                                            kv_cache_spec.dtypes):
                    dtype_size = get_dtype_size(dtype)
                    num_element_per_page = (
                        kv_cache_spec.page_size_bytes // dtype_size)
                    target_shape = (num_blocks, *shape)
                    stride = torch.empty(target_shape).stride()
                    target_stride = (num_element_per_page, *stride[1:])
                    assert storage_offset_bytes % dtype_size == 0
                    tensor = torch.as_strided(
                        raw_tensor.view(dtype),
                        size=target_shape,
                        stride=target_stride,
                        storage_offset=storage_offset_bytes // dtype_size,
                    )
                    state_tensors.append(tensor)
                    storage_offset_bytes += stride[0] * dtype_size
                kv_caches[layer_name] = state_tensors
            else:
                raise ValueError("Unknown KV cache spec type.")

    bind_kv_cache(kv_caches,
                    self.compilation_config.static_forward_context,
                    self.kv_caches)

    return kv_caches

# apply patch
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
NPUModelRunner.initialize_kv_cache_tensors = _initialize_kv_cache_tensors
