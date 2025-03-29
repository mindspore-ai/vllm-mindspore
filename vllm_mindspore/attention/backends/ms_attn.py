#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Attention layer with MsAttention."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import torch

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    AttentionState,
    AttentionLayer,
)

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUBuilder

from vllm.utils import make_tensor_with_pad
from vllm.attention.backends.utils import (
    compute_slot_mapping,
    compute_slot_mapping_start_idx,
    is_block_tables_empty,
)
from vllm.multimodal import MultiModalPlaceholderMap

from vllm_mindspore.attention.backends.utils import MsAttentionState
from vllm_mindspore.attention.ops.paged_attn import PagedAttentionMetadata

from vllm_mindspore.utils import MsKVCache

import mindspore as ms
from mindspore import mutable
from mindspore._c_expression import swap_cache

def advance_step_op(sampled_token_ids,
                    model_input,
                    seq_lens_tensor,
                    num_queries,
                    block_size,
                    block_tables,
                    slot_mapping):
    # update input_tokens
    sampled_token_ids_list = sampled_token_ids[:
                                               num_queries].squeeze(  # type: ignore
                                                   -1)
    model_input.input_tokens[:
                             num_queries] = sampled_token_ids_list  # type: ignore

    # get seq_lens and input_positions
    seq_lens = seq_lens_tensor[:num_queries]
    next_seq_lens = seq_lens + 1
    next_input_pos = next_seq_lens - 1

    # update seq_lens and input_positions
    seq_lens_tensor[:num_queries] = next_seq_lens
    model_input.input_positions[:
                                num_queries] = next_input_pos  # type: ignore

    # 计算 block index 和 offset
    block_idx = next_input_pos // block_size
    block_offset = next_input_pos % block_size

    current_block_table = block_tables.gather(
        1, block_idx.unsqueeze(-1)).squeeze(-1)
    slot_num = current_block_table * block_size + block_offset

    # update slot_mapping
    slot_mapping[:num_queries] = slot_num


@dataclass
class MSAttentionMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for TorchSDPABackend."""

    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    chunked_prefill: bool
    seq_lens: Optional[List[int]] = None  # For non-chunked prefill

    # For chunked prefill only
    max_query_len: Optional[int] = None

    max_prefill_seq_len: int = 0
    seq_start_loc: Optional[torch.Tensor] = None
    _cached_prefill_metadata: Optional["MSAttentionMetadata"] = None
    _cached_decode_metadata: Optional["MSAttentionMetadata"] = None
    context_lens_tensor: Optional[torch.Tensor] = None
    encoder_seq_start_loc: Optional[torch.Tensor] = None
    max_decode_query_len: Optional[int] = None

    max_kv_len: Optional[int] = None
    query_start_loc: Optional[torch.Tensor] = None
    kv_start_loc: Optional[torch.Tensor] = None
    prefill_block_tables: Optional[torch.Tensor] = None
    query_lens: Optional[List[int]] = None

    # Begin encoder attn & enc/dec cross-attn fields...
    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None

    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: Optional[int] = None

    # Number of tokens input to encoder
    num_encoder_tokens: Optional[int] = None

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    use_cuda_graph: bool = False
    enable_kv_scales_calculation: bool


    @property
    def prefill_metadata(self):
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert ((self.seq_lens is not None)
                or (self.encoder_seq_lens is not None))
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        query_start_loc = (None if self.query_start_loc is None else
                           self.query_start_loc[:self.num_prefills + 1])
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[:self.num_prefill_tokens])
        seq_lens = (None if self.seq_lens is None else
                    self.seq_lens[:self.num_prefills])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[:self.num_prefills])
        seq_start_loc = (None if self.seq_start_loc is None else
                         self.seq_start_loc[:self.num_prefills + 1])
        context_lens_tensor = (None if self.context_lens_tensor is None else
                               self.context_lens_tensor[:self.num_prefills])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[:self.num_prefills])

        self._cached_prefill_metadata = MSAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=self.
            multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation=False,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_query_len=0,
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            encoder_seq_start_loc=self.encoder_seq_start_loc,
            max_encoder_seq_len=self.max_encoder_seq_len,
            chunked_prefill=self.chunked_prefill,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables)
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self):
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[self.num_prefill_tokens:])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[self.num_prefills:])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[self.num_prefills:])

        self._cached_decode_metadata = MSAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
            seq_lens=None,
            seq_lens_tensor=seq_lens_tensor,
            max_decode_query_len=self.max_decode_query_len,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            # Batch may be composed of prefill|decodes, adjust query start
            # indices to refer to the start of decodes. E.g.
            # in tokens:[3 prefills|6 decodes], query_start_loc=[3,9] => [0,6].
            query_start_loc=(self.query_start_loc[self.num_prefills:] -
                             self.query_start_loc[self.num_prefills])
            if self.query_start_loc is not None else None,
            seq_start_loc=self.seq_start_loc[self.num_prefills:]
            if self.seq_start_loc is not None else None,
            context_lens_tensor=None,
            block_tables=block_tables,
            use_cuda_graph=self.use_cuda_graph,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            encoder_seq_start_loc=self.encoder_seq_start_loc,
            max_encoder_seq_len=self.max_encoder_seq_len,
            chunked_prefill=self.chunked_prefill,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables)
        return self._cached_decode_metadata

    def advance_step(self,
                     model_input: "ModelInputForNPUWithSamplingMetadata",
                     sampled_token_ids: Optional[torch.Tensor],
                     block_size: int,
                     num_seqs: int,
                     num_queries: int,
                     turn_prefills_into_decodes: bool = False):
        """
        Update metadata in-place to advance one decode step.
        """
        # When using cudagraph, the num_seqs is padded to the next captured
        # batch sized, but num_queries tracks the actual number of requests in
        # the batch. For --enforce-eager mode, num_seqs == num_queries
        if num_seqs != num_queries:
            assert num_seqs > num_queries

        if turn_prefills_into_decodes:
            # When Mutli-Step is enabled with Chunked-Prefill, prefills and
            # decodes are scheduled together. In the first step, all the
            # prefills turn into decodes. This update reflects that
            # conversion.
            assert self.num_decode_tokens + self.num_prefills == num_seqs
            self.num_decode_tokens += self.num_prefills
            self.num_prefills = 0
            self.num_prefill_tokens = 0
            self.max_prefill_seq_len = 0
            self.max_query_len = 1

            self.slot_mapping = self.slot_mapping[:num_seqs]
        else:
            assert self.seq_lens is not None
            assert self.max_decode_seq_len == max(self.seq_lens)

        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.num_decode_tokens == num_seqs
        assert self.slot_mapping.shape == (num_seqs, )

        assert self.seq_lens is not None
        assert len(self.seq_lens) == num_seqs
        assert self.seq_lens_tensor is not None
        assert self.seq_lens_tensor.shape == (num_seqs, )
        assert self.max_query_len == 1
        assert self.max_prefill_seq_len == 0

        assert self.query_start_loc is not None
        assert self.query_start_loc.shape == (num_queries + 1, )
        assert self.seq_start_loc is not None
        assert self.seq_start_loc.shape == (num_seqs + 1, )

        assert self.context_lens_tensor is not None
        assert self.context_lens_tensor.shape == (num_queries, )

        assert self.block_tables is not None
        assert self.block_tables.shape[0] == num_seqs

        # Update query lengths. Note that we update only queries and not seqs,
        # since tensors may be padded due to captured cuda graph batch size
        for i in range(num_queries):
            self.seq_lens[i] += 1
        self.max_decode_seq_len = max(self.seq_lens)

        advance_step_op(sampled_token_ids,
                        model_input,
                        self.seq_lens_tensor,
                        num_queries,
                        block_size,
                        self.block_tables,
                        self.slot_mapping)

    def get_seq_lens(
        self,
        attn_type: str,
    ):
        """
        Extract appropriate sequence lengths from attention metadata
        according to attention type.

        Arguments:

        * attn_metadata: Attention metadata structure associated with attention
        * attn_type: encoder attention, decoder self-attention,
                    encoder/decoder cross-attention

        Returns:
        * Appropriate sequence lengths tensor for query
        * Appropriate sequence lengths tensor for key & value
        """

        if (
            attn_type == AttentionType.DECODER
            or attn_type == AttentionType.ENCODER_ONLY
        ):
            seq_lens_q = self.seq_lens
            seq_lens_kv = self.seq_lens
        elif attn_type == AttentionType.ENCODER:
            seq_lens_q = self.encoder_seq_lens
            seq_lens_kv = self.encoder_seq_lens
        elif attn_type == AttentionType.ENCODER_DECODER:
            seq_lens_q = self.seq_lens
            seq_lens_kv = self.encoder_seq_lens
        else:
            raise AttributeError(f"Invalid attention type {str(attn_type)}")
        return seq_lens_q, seq_lens_kv

    def get_seq_len_block_table_args(
        self,
        attn_type: str,
    ) -> tuple:
        if (
            attn_type == AttentionType.DECODER
            or attn_type == AttentionType.ENCODER_ONLY
        ):
            # Decoder self-attention
            # Choose max_seq_len based on whether we are in prompt_run
            return (self.seq_lens_tensor, self.max_decode_seq_len, self.block_tables)
        elif attn_type == AttentionType.ENCODER_DECODER:
            # Enc/dec cross-attention KVs match encoder sequence length;
            # cross-attention utilizes special "cross" block tables
            return (
                self.encoder_seq_lens_tensor,
                self.max_encoder_seq_len,
                self.cross_block_tables,
            )
        elif attn_type == AttentionType.ENCODER:
            # No block tables associated with encoder attention
            return (self.encoder_seq_lens_tensor, self.max_encoder_seq_len, None)
        else:
            raise AttributeError(f"Invalid attention type {str(attn_type)}")

    def keys(self):
        return ["num_prefill_tokens", "num_decode_tokens", "slot_mapping", "batch_valid_length", "context_lens", "block_tables"]

    def __getitem__(self, key):
        if key == "context_lens":
            key = "seq_lens_tensor"
        if key == "batch_valid_length":
            return mutable(getattr(self, "seq_lens"), dynamic_len=True)
        if key == "block_tables":
            if getattr(self, key).ndim == 1:
                return mutable(getattr(self, key).expand_dims(0))
            return mutable(getattr(self, key))
        return mutable(getattr(self, key))

class MsAttentionMetadataBuilder(AttentionMetadataBuilder[MSAttentionMetadata]):

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        self.input_builder = input_builder
        self.runner = input_builder.runner
        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size

    def prepare(self):
        self.slot_mapping: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.curr_seq_lens: List[int] = []
        self.multimodal_placeholder_maps: Dict[str, MultiModalPlaceholderMap] = (
            defaultdict(MultiModalPlaceholderMap)
        )
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0
        self.has_prefix_cache_hit = False


    def _add_seq_group(
        self,
        inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
        chunked_prefill_enabled: bool,
        prefix_cache_hit: bool,
    ):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables

        for (
            seq_id,
            token_len,
            seq_len,
            curr_seq_len,
            query_len,
            context_len,
            curr_sliding_window_block,
        ) in zip(
            inter_data.seq_ids,
            [len(t) for t in inter_data.input_tokens],
            inter_data.orig_seq_lens,
            inter_data.seq_lens,
            inter_data.query_lens,
            inter_data.context_lens,
            inter_data.curr_sliding_window_blocks,
        ):
            self.context_lens.append(context_len)

            if is_prompt:
                mm_maps = inter_data.multi_modal_placeholder_maps
                if mm_maps:
                    for modality, placeholders in mm_maps.items():
                        self.multimodal_placeholder_maps[modality].extend(placeholders)

                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if prefix_cache_hit:
                # NOTE(woosuk): For flash-attn, the block table should
                # include the entries for the incoming prefill tokens.
                block_table = block_tables[seq_id]
            elif (
                chunked_prefill_enabled or not is_prompt
            ) and block_tables is not None:
                if curr_sliding_window_block == 0:
                    block_table = block_tables[seq_id]
                else:
                    block_table = block_tables[seq_id][-curr_sliding_window_block:]
            self.block_tables.append(block_table)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(
                is_prompt, query_len, context_len, self.sliding_window
            )
            compute_slot_mapping(
                is_profile_run,
                self.slot_mapping,
                seq_id,
                seq_len,
                context_len,
                start_idx,
                self.block_size,
                inter_data.block_tables,
            )

    def build(
        self,
        seq_lens: List[int],
        query_lens: List[int],
        cuda_graph_pad_size: int,
        batch_size: int,
    ):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        prefix_cache_hit = any(
            [
                inter_data.prefix_cache_hit
                for inter_data in self.input_builder.inter_data_list
            ]
        )
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(
                inter_data, self.input_builder.chunked_prefill_enabled, prefix_cache_hit
            )

        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        max_query_len = max(query_lens)
        decode_query_lens = query_lens[self.num_prefills:]
        if len(decode_query_lens) > 0:
            max_decode_query_len = max(decode_query_lens)
        else:
            max_decode_query_len = 1
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens
        query_start_loc = list(accumulate(query_lens, initial=0))
        seq_start_loc = list(accumulate(seq_lens, initial=0))

        if use_captured_graph:
            raise RuntimeError("Doesnot support captured graph now!")
        else:
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=-1,
                dtype=torch.int,
                device=device,
            )
        assert max_query_len > 0, "query_lens: {}".format(query_lens)

        context_lens_tensor = ms.Tensor(self.context_lens, dtype=ms.int32)
        seq_lens_tensor = ms.Tensor(seq_lens, dtype=ms.int32)

        slot_mapping_tensor = ms.Tensor(self.slot_mapping, dtype=ms.int32)
        query_start_loc_tensor = ms.Tensor(query_start_loc, dtype=ms.int32)
        seq_start_loc_tensor = ms.Tensor(seq_start_loc, dtype=ms.int32)

        return MSAttentionMetadata(
            slot_mapping=slot_mapping_tensor,
            block_tables=block_tables,
            seq_lens_tensor=seq_lens_tensor,
            seq_lens=seq_lens,
            max_decode_seq_len=max_decode_seq_len,
            chunked_prefill=self.input_builder.chunked_prefill_enabled,
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
            query_lens=query_lens,
            query_start_loc=query_start_loc_tensor,
            seq_start_loc=seq_start_loc_tensor,
            context_lens_tensor=context_lens_tensor,
            max_query_len=max_query_len,
        )


class MsAttentionBackend(AttentionBackend):
    """MindSpore attention backend."""

    @staticmethod
    def get_name() -> str:
        return "MS_ATTN"

    @staticmethod
    def get_impl_cls() -> Type["AttentionImpl"]:
        return MsAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return MSAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["MsAttentionMetadataBuilder"]:
        return MsAttentionMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["AttentionState"]:
        return MsAttentionState

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
    def swap_blocks(
        src_kv_cache: MsKVCache,
        dst_kv_cache: MsKVCache,
        src_to_dst: torch.Tensor,
        swap_type: bool,
    ) -> None:
        """
        Swap key/value cache between host and device, to support multi-batch and long-sequence inference.

        Args:
            src_kv_cache: Source KV cache block.
            dst_kv_cache: Destination KV cache block.
            src_to_dst: A 2-D array contains src and dst blocks to swap.
            swap_type: A bool value indicating the data direction: "True" for device-to-host, and "False" for host-to-device.
        """
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        swap_cache(src_key_cache, dst_key_cache, src_to_dst, swap_type)
        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        swap_cache(src_value_cache, dst_value_cache, src_to_dst, swap_type)

    @staticmethod
    def copy_blocks(
        kv_caches: List[MsKVCache],
        src_to_dists: torch.Tensor,
    ) -> None:
        blocks_to_copy = src_to_dists.asnumpy().tolist()
        for kv_cache in kv_caches:
            npu_key_block, npu_value_block = kv_cache
            for src, dst in blocks_to_copy:
                npu_key_block[dst, :] = npu_key_block[src, :]
                npu_value_block[dst, :] = npu_value_block[src, :]


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
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        pass

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MSAttentionMetadata,
        attn_type: str = AttentionType.DECODER,
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


class MLABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "MS_MLA"

    @staticmethod
    def get_impl_cls() -> Type["AttentionImpl"]:
        return MsAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return MSAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["MsAttentionMetadataBuilder"]:
        return MsAttentionMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["AttentionState"]:
        return MsAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
    ) -> Tuple[int, ...]:
        return (1, num_blocks, block_size, 1, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        swap_cache(src_key_cache, dst_key_cache, src_to_dst)


    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        blocks_to_copy = src_to_dists.asnumpy().tolist()
        for kv_cache in kv_caches:
            npu_key_block = kv_cache[0]
            for src, dst in blocks_to_copy:
                npu_key_block[dst, :] = npu_key_block[src, :]

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [576]
