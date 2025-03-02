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

from typing import Optional, Type

import torch
from vllm.attention.backends.abstract import AttentionBackend

from functools import lru_cache
from typing import Optional, Type

import torch

import vllm.envs as envs
from vllm.attention.backends.abstract import AttentionBackend
from vllm.logger import init_logger
from vllm.platforms import _Backend, current_platform

logger = init_logger(__name__)


def which_attn_to_use(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: Optional[str],
    block_size: int,
    is_attention_free: bool,
    use_v1: bool = False,
    use_mla: bool = False,
):
    """Returns which flash attention backend to use."""
    if use_mla:
        return "MLA_ATTN"

    selected_backend = _Backend.FLASH_ATTN

    if is_attention_free:
        return _Backend.NO_ATTENTION

    backend_by_env_var: Optional[str] = envs.VLLM_ATTENTION_BACKEND
    if backend_by_env_var is not None:
        logger.warning(
            "MindSpore donot support %s attention backend now!"
            % str(backend_by_env_var)
        )

    # get device-specific default attn_backend
    default_backend = current_platform.get_default_attn_backend(selected_backend)
    if default_backend is not None:
        return default_backend

    return selected_backend


@lru_cache(maxsize=None)
def _cached_get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: Optional[str],
    block_size: int,
    is_attention_free: bool,
    is_blocksparse: bool = False,
    use_v1: bool = False,
    use_mla: bool = False,
) -> Type[AttentionBackend]:
    if is_blocksparse:
        logger.warning(
            "MindSpore doesnot support BlocksparseFlashAttention backend now."
        )

    backend = which_attn_to_use(
        head_size, dtype, kv_cache_dtype, block_size, is_attention_free, use_v1, use_mla
    )
    if backend == _Backend.FLASH_ATTN:
        logger.info("Using Flash Attention backend.")
        from vllm_mindspore.attention.backends.ms_attn import (  # noqa: F401
            MsAttentionBackend,
        )

        return MsAttentionBackend
    elif backend == "MLA_ATTN":
        logger.info("Using ML Attention backend.")
        from vllm_mindspore.attention.backends.ms_attn import MLABackend

        return MLABackend
    else:
        raise ValueError("Invalid attention backend.")


def get_ms_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: Optional[str],
    block_size: int,
    is_attention_free: bool,
    is_blocksparse: bool = False,
    use_mla: bool = False,
) -> Type[AttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""
    # Accessing envs.* behind an @lru_cache decorator can cause the wrong
    # value to be returned from the cache if the value changes between calls.
    # To avoid this, we read envs.VLLM_USE_V1 here and pass it explicitly to the
    # private function.
    return _cached_get_attn_backend(
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        block_size=block_size,
        is_attention_free=is_attention_free,
        is_blocksparse=is_blocksparse,
        use_v1=envs.VLLM_USE_V1,
        use_mla=use_mla,
    )
