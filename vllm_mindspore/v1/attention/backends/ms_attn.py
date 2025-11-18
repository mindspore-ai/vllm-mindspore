# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/v1/attention/backends/flash_attn.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2024-2025 The vLLM team.
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
"""Attention layer with MsAttention."""
from dataclasses import dataclass
from typing import Any, Optional

import mindspore as ms
import numpy as np
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import (AttentionCGSupport,
                                              CommonAttentionMetadata)

logger = init_logger(__name__)


class MsAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "MS_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return MsAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return MsAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["MsAttentionMetadataBuilder"]:
        return MsAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


class MLABackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "MS_MLA"

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return MsAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return MsAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["MsAttentionMetadataBuilder"]:
        return MsAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (1, num_blocks, block_size, 1, head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [576]


@dataclass
class MsAttentionMetadata:
    """
    AttentionMetadata for vllm-mindspore V1
    """
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    # add by vllm-mindspore begin
    seq_lens_np: np.ndarray
    block_tables: ms.Tensor
    q_seq_lens_np: np.ndarray
    context_lens: ms.Tensor
    max_context_lens: int
    # add by vllm-mindspore end

    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    seq_lens: ms.Tensor
    slot_mapping: ms.Tensor

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.


class MsAttentionImpl(AttentionImpl):
    """
    AttentionImpl for vllm-mindspore V1
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> None:
        pass

    def forward(
        self,
        layer: ms.nn.Cell,
        query: ms.Tensor,
        key: ms.Tensor,
        value: ms.Tensor,
        kv_cache: ms.Tensor,
        attn_metadata: MsAttentionMetadata,
        output: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        """Forward pass with MsAttention.
        """
        pass


class MsAttentionMetadataBuilder:
    cudagraph_support = AttentionCGSupport.NEVER
    reorder_batch_threshold = None

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config)
        self.num_heads_kv = self.model_config.get_num_kv_heads(
            self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.max_num_splits = 0  # No upper bound on the number of splits.
        self.aot_schedule = False

        self.use_full_cuda_graph = False

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: Optional[tuple[int, int]] = None

    def reorder_batch(self, input_batch, scheduler_output) -> bool:
        return False

    def build(self,
              common_prefix_len: int,
              common_attn_metadata,
              fast_build: bool = False):
        """
        fast_build disables AOT scheduling, used when there will be few 
        iterations i.e. spec-decode
        """
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = ms.from_numpy(common_attn_metadata.slot_mapping_np)

        seq_lens_np = common_attn_metadata.seq_lens_np
        num_computed_tokens_np = common_attn_metadata.num_computed_tokens_np
        max_context_lens = num_computed_tokens_np.max()
        context_lens = ms.from_numpy(num_computed_tokens_np)

        q_seq_lens_np = common_attn_metadata.q_seq_lens_np

        attn_metadata = MsAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            seq_lens_np=seq_lens_np,
            q_seq_lens_np=q_seq_lens_np,
            context_lens=context_lens,
            max_context_lens=max_context_lens,
            block_tables=block_table_tensor,
            slot_mapping=slot_mapping,
        )
        return attn_metadata


FlashAttentionMetadata = MsAttentionMetadata


@dataclass
class MsCommonAttentionMetadata(CommonAttentionMetadata):
    q_seq_lens_np: np.ndarray = None
    seq_lens_np: np.ndarray = None
    num_computed_tokens_np: np.ndarray = None
    slot_mapping_np: np.ndarray = None
