# SPDX-License-Identifier: Apache-2.0
"""Attention layer with FlashAttention."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.logger import init_logger


from vllm_mindspore.utils import MsKVCache

import mindspore as ms
from mindspore import mutable
from mindspore._c_expression import swap_cache


logger = init_logger(__name__)


class FlashAttentionBackend(AttentionBackend):

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
        return FlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["AttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

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
        return FlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["AttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

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
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    max_seq_len: int
    seq_lens: torch.Tensor
    seq_lens_np: np.ndarray
    block_tables: torch.Tensor
    slot_mapping: torch.Tensor
    q_seq_lens: torch.Tensor
    q_seq_lens_np: np.ndarray
    context_lens: torch.Tensor
    max_context_lens: int
    query_start_loc: torch.Tensor

    def __getitem__(self, key):
        if key == "batch_valid_length":
            key = "seq_lens"
        return getattr(self, key)


class MsAttentionImpl(AttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:
    |<----------------- num_decode_tokens ------------------>|
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
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
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            output: shape = [num_tokens, num_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run.
            attn_metadata: Metadata for attention.
        NOTE: It in-place updates the output tensor.
        """
        pass


class FlashAttentionMetadataBuilder:
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
        q_seq_lens = ms.from_numpy(q_seq_lens_np)

        attn_metadata = FlashAttentionMetadata(
            seq_lens=seq_lens,
            seq_lens_np=seq_lens_np,
            block_tables=(self.runner.input_batch.block_table.get_device_tensor()[:num_reqs]),
            slot_mapping=slot_mapping,
            q_seq_lens=q_seq_lens,
            q_seq_lens_np=q_seq_lens_np,
            max_seq_len=max_seq_len,
            context_lens=context_lens,
            max_context_lens=max_context_lens,
            query_start_loc = query_start_loc
        )
        return attn_metadata
