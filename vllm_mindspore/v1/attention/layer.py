# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/attention/layer.py
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
"""Common layer for LLM."""
from typing import Any, Optional

from mindspore import Tensor, nn, ops
from mindspore.ops.auto_generate import PagedAttention, ReshapeAndCache
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from vllm.attention.backends.abstract import AttentionType
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

from vllm_mindspore.model_executor.utils import get_model_context


class Attention(nn.Cell):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 scale: float,
                 num_kv_heads: Optional[int] = None,
                 alibi_slopes: Optional[list[float]] = None,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 blocksparse_params: Optional[dict[str, Any]] = None,
                 logits_soft_cap: Optional[float] = None,
                 per_layer_sliding_window: Optional[int] = None,
                 use_mla: bool = False,
                 prefix: str = "",
                 attn_type: str = AttentionType.DECODER,
                 **extra_impl_args) -> None:
        super().__init__()
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Only support DECODER now.")
        if not num_kv_heads:
            num_kv_heads = num_heads
        self.attn_type = attn_type
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.hidden_size_per_partition = num_heads * head_size
        self.kv_hidden_size_per_partition = num_kv_heads * head_size

        input_layout = "TH"
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
        self.is_eager_mode = (
            get_current_vllm_config().model_config.enforce_eager)

    def construct(self, query: Tensor, key: Tensor, value: Tensor,
                  key_cache: Tensor, value_cache: Tensor, slot_mapping: Tensor,
                  attn_mask: Tensor, batch_valid_length: Tensor,
                  q_seq_lens: Tensor, block_tables: Tensor) -> Tensor:
        """Attention forward, support MHA and GQA.

        Args:
            query: shape = [1, num_tokens, hidden_size]
            key: shape = [1, num_tokens, hidden_size]
            value: shape = [1, num_tokens, hidden_size]
            ...
            slot_mapping: shape = [seq_length, ]
            batch_valid_length: shape = [batch_size, ]
            block_tables: shape = [block_size, num_block]
        """
        if self.is_eager_mode:
            # ensure that the input tensors of reshape_and_cache is continuous
            key = key.contiguous()
            value = value.contiguous()

        cache_out = self.reshape_and_cache(key, value, key_cache, value_cache,
                                           slot_mapping)
        query = ops.depend(query, cache_out)
        if get_model_context("is_prefill"):
            output = self._run_prefill_forward(query, key, value, attn_mask,
                                               batch_valid_length,
                                               batch_valid_length)
        else:
            output = self._run_decode_forward(query, key_cache, value_cache,
                                              block_tables, batch_valid_length,
                                              attn_mask, q_seq_lens)
        return output

    def _run_prefill_forward(self, query: Tensor, key: Tensor, value: Tensor,
                             attn_mask: Tensor, actual_seq_qlen: tuple[int],
                             actual_seq_kvlen: tuple[int]) -> Tensor:
        """Prefill with FlashAttention.

        Args:
            query: shape = [1, num_tokens, hidden_size]
            key: shape = [1, num_tokens, hidden_size]
            value: shape = [1, num_tokens, hidden_size]
            actual_seq_qlen: shape = [batch_size, ]
            actual_seq_kvlen: shape = [batch_size, ]

        NOTE: Currently `PyNative` mode does not support operations in "TH"
              form, so it will be converted to "BSH" form.
        """
        _, _, _, output = self.flash_attention(query, key, value, None, None,
                                               None, attn_mask, None,
                                               actual_seq_qlen,
                                               actual_seq_kvlen)
        return output

    def _run_decode_forward(self, query: Tensor, key_cache: Tensor,
                            value_cache: Tensor, block_tables: Tensor,
                            batch_valid_length: Tensor, attn_mask: Tensor,
                            q_seq_lens: Tensor) -> Tensor:
        """Decode with PagedAttention.

        Args:
            query: Its shape is [batch_size, 1, hidden_size].
            key_cache: Its shape is
                [num_block, block_size, kv_heads_per_partition, head_size].
            value_cache: Its shape is
                [num_block, block_size, kv_heads_per_partition, head_size].
            block_tables: Its shape is [block_size, num_block].
            context_lens: Its shape is [batch_size, ].
        """
        output = self.paged_attention(query, key_cache, value_cache,
                                      block_tables, batch_valid_length, None,
                                      None, attn_mask, q_seq_lens)
        return output
