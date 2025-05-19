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
"""Common layer for LLM."""
from typing import Any, Dict, List, Optional, Tuple

from mindspore import Tensor, mint, nn, ops, jit
from mindspore.common import dtype as mstype
from mindspore.ops.auto_generate import PagedAttention, ReshapeAndCache
from mindspore.ops.operations.nn_ops import FlashAttentionScore

from vllm.config import CacheConfig
from vllm.attention.backends.abstract import AttentionType
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)


def _pad_to_max_tensor(
        input_: Tensor,
        max_len: int,
        dim: int = 0,
        pad_value: int = -1
) -> Tensor:
    """Temporary function, will be deprecated in the future."""
    if input_.shape[dim] == max_len:
        return input_
    pad_shape = (input_.shape[0], max_len - input_.shape[dim], *input_.shape[dim + 1:])
    pad_tensor = mint.ones(size=pad_shape, dtype=input_.dtype) * pad_value
    output = mint.cat([input_, pad_tensor], dim=dim)
    return output


def _generate_attn_mask(
    query: Tensor,
    value: Tensor,
    flatten: bool
) -> Tensor:
    """Temporary function, will be deprecated in the future."""
    if flatten:
        return mint.triu(mint.ones(size=(128, 128), dtype=query.dtype), 1)
    q_seq_len = query.shape[1]
    kv_seq_len = value.shape[1]
    mask = mint.ones((q_seq_len, kv_seq_len), dtype=mstype.uint8)
    mask = mint.triu(mask, diagonal=1)
    return mask


def _hidden_states_th2bsh(
    input_: Tensor,
    batch_valid_length: Tensor
) -> Tensor:
    """Temporary function, will be deprecated in the future."""
    max_seq_len = batch_valid_length.max().item()
    start_pos = 0
    padding_input_list = []
    for valid_length in batch_valid_length:
        valid_input = input_[:, start_pos: start_pos + valid_length, :]
        padded_input = _pad_to_max_tensor(valid_input, max_seq_len, 1)
        padding_input_list.append(padded_input)
        start_pos += valid_length
    bsh_output = mint.cat(padding_input_list, dim=0)
    return bsh_output


def _hidden_states_bsh2th(
    input_: Tensor,
    batch_valid_length: Tensor
) -> Tensor:
    """Temporary function, will be deprecated in the future."""
    unpadded_input_list = []
    for batch_index, valid_length in enumerate(batch_valid_length):
        padded_input = input_[batch_index:batch_index + 1]
        unpadded_input = padded_input[:, :valid_length, ...]
        unpadded_input_list.append(unpadded_input)
    th_output = mint.cat(unpadded_input_list, dim=1)
    return th_output


class Attention(nn.Cell):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        per_layer_sliding_window: Optional[int] = None,
        use_mla: bool = False,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Only support DECODER now.")
        if not num_kv_heads:
            num_kv_heads = num_heads
        self.attn_type = attn_type
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.hidden_size_per_partition = num_heads*head_size
        self.kv_hidden_size_per_partition = num_kv_heads*head_size
        self.flatten = True

        input_layout = "TH" if self.flatten else "BSH"  # pynative 下不支持拉平操作。
        scale = float(scale)
        pre_tokens = 2147483647
        next_tokens = 2147483647

        self.reshape_and_cache = ReshapeAndCache()
        self.flash_attention = FlashAttentionScore(head_num=num_heads,
                                                   scale_value=scale,
                                                   pre_tokens=pre_tokens,
                                                   next_tokens=next_tokens,
                                                   input_layout=input_layout)
        self.paged_attention = PagedAttention(head_num=num_heads,
                                              scale_value=scale,
                                              kv_head_num=num_kv_heads)

    @jit
    def construct(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        is_prefill: bool,
        slot_mapping: Tensor,
        attn_mask: Tensor,
        batch_valid_length: Tensor,
        q_seq_lens: Tensor,
        block_tables: Tensor,
    ) -> Tensor:
        """Attention foward, support MHA and GQA.

        Args:
            query: shape = [1, num_tokens, hidden_size]
            key: shape = [1, num_tokens, hidden_size]
            value: shape = [1, num_tokens, hidden_size]
            ...
            slot_mapping: shape = [seq_length, ]
            batch_valid_length: shape = [batch_size, ]
            block_tables: shape = [block_size, num_block]
        """
        output = query
        cache_out = self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
        query = ops.depend(query, cache_out)
        if is_prefill:
            output = self._run_prefill_forward(query, key, value, attn_mask, batch_valid_length, batch_valid_length)
        else:
            output = self._run_decode_forward(query, key_cache, value_cache, block_tables, batch_valid_length,
                                              attn_mask, q_seq_lens)
        return output

    def _run_prefill_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor,
        actual_seq_qlen: Tuple[int],
        actual_seq_kvlen: Tuple[int],
    ) -> Tensor:
        """Prefill with FlashAttention.

        Args:
            query: shape = [1, num_tokens, hidden_size]
            key: shape = [1, num_tokens, hidden_size]
            value: shape = [1, num_tokens, hidden_size]
            actual_seq_qlen: shape = [batch_size, ]
            actual_seq_kvlen: shape = [batch_size, ]
        NOTE: Currently `PyNative` mode does not support operations in "TH" form, so it will be converted to "BSH" form.
        """
        query = query.view(-1, self.hidden_size_per_partition)
        key = key.view(-1, self.kv_hidden_size_per_partition)
        value = value.view(-1, self.kv_hidden_size_per_partition)
        _, _, _, output = self.flash_attention(
            query,
            key,
            value,
            None,
            None,
            None,
            attn_mask,
            None,
            actual_seq_qlen,
            actual_seq_kvlen
        )
        output = output.view(1, -1, self.hidden_size_per_partition)
        return output

    def _run_decode_forward(
        self,
        query: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        block_tables: Tensor,
        batch_valid_length: Tensor,
        attn_mask: Tensor,
        q_seq_lens: Tensor,
    ) -> Tensor:
        """Decode with PagedAttention.

        Args:
            query: shape = [batch_size, 1, hidden_size]
            key_cache: shape = [num_block, block_size, kv_heads_per_partition, head_size]
            value_cache: shape = [num_block, block_size, kv_heads_per_partition, head_size]
            block_tables: shape = [block_size, num_block]
            context_lens: shape = [batch_size, ]
        """
        output = self.paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            batch_valid_length,
            None,
            None,
            attn_mask,
            q_seq_lens
        )
        return output
