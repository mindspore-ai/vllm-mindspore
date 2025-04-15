# SPDX-License-Identifier: Apache-2.0
"""Attention layer with FlashAttention."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.logger import init_logger


from vllm_mindspore.utils import MsKVCache

import mindspore as ms
from mindspore import mutable
from mindspore._c_expression import swap_cache


logger = init_logger(__name__)


class MsAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "MS_ATTN"

    @staticmethod
    def get_impl_cls() -> Type["AttentionImpl"]:
        return MsAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return MsAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["MsAttentionMetadataBuilder"]:
        return MsAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
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
    def get_impl_cls() -> Type["AttentionImpl"]:
        return MsAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return MsAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["MsAttentionMetadataBuilder"]:
        return MsAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (1, num_blocks, block_size, 1, head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
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

    #num_actual_tokens: int = None  # Number of tokens excluding padding.
    #max_query_len: int 
    query_start_loc: ms.Tensor
    max_seq_len: int
    seq_lens: ms.Tensor
    #block_table: torch.Tensor
    slot_mapping: ms.Tensor

    # For cascade attention.
    #use_cascade: bool
    #common_prefix_len: int
    #cu_prefix_query_lens: Optional[torch.Tensor]
    #prefix_kv_lens: Optional[torch.Tensor]
    #suffix_kv_lens: Optional[torch.Tensor]

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
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
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
    def __init__(self, runner: "GPUModelRunner"):
        self.runner = runner

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(self, num_reqs: int, num_actual_tokens: int, max_query_len: int,
              common_prefix_len: int):
        # do not manually call 'tensor.move_to("Ascend", blocking=False)' here,
        # because it will cause a certain amount of host time.
        query_start_loc = ms.from_numpy(self.runner.query_start_loc_np[:num_reqs + 1])
        max_context_lens = self.runner.input_batch.num_computed_tokens_cpu[:num_reqs].max()
        slot_mapping = ms.from_numpy(self.runner.slot_mapping_np[:num_actual_tokens])
        seq_lens_np = self.runner.seq_lens_np[:num_reqs]
        max_seq_len = seq_lens_np.max()
        seq_lens = ms.from_numpy(seq_lens_np)
        context_lens = ms.from_numpy(self.runner.input_batch.num_computed_tokens_cpu[:num_reqs])

        q_seq_lens_np = np.diff(self.runner.query_start_loc_np[:num_reqs + 1])

        attn_metadata = MsAttentionMetadata(
            seq_lens=seq_lens,
            seq_lens_np=seq_lens_np,
            block_tables=(self.runner.input_batch.block_table.get_device_tensor()[:num_reqs]),
            slot_mapping=slot_mapping,
            q_seq_lens_np=q_seq_lens_np,
            max_seq_len=max_seq_len,
            context_lens=context_lens,
            max_context_lens=max_context_lens,
            query_start_loc = query_start_loc
        )
        return attn_metadata

FlashAttentionMetadata = MsAttentionMetadata