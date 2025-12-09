# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/model_executor/layers/quantization/kv_cache.py
#
#
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

from typing import Optional

import mindspore
import numpy as np
from mindspore import Parameter, Tensor, dtype, ops
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.ops.auto_generate import PagedAttention, ReshapeAndCache
from mindspore.ops.operations._infer_ops import QuantV2
from mindspore.ops.operations.nn_ops import (FlashAttentionScore,
                                             PromptFlashAttention)
from vllm.attention.backends.abstract import AttentionType
from vllm.config import CacheConfig
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs

from vllm_mindspore.model_executor.utils import get_model_context
from vllm_mindspore.utils import is_310p


class BaseKVCacheMethod(QuantizeMethodBase):
    """
    Quant method that adds `_k_scale` and `_v_scale` attributes to the
    Attention layer to support loading those scaling factors from checkpoints.
    The k/v_scale will be used to:
        - quantize k/v_cache entries before saving them to the cache
        - dequantize k/v_cache entries before fetching them from the cache

    :param quant_config: the appropriate QuantizationConfig
    """

    def __init__(self, quant_config=None):
        self.quant_config = quant_config

    def create_weights(self,
                       layer: mindspore.nn.Cell,
                       num_heads: int,
                       head_size: int,
                       scale: float,
                       num_kv_heads: Optional[int] = None,
                       use_pfa: bool = False,
                       cache_config: CacheConfig = None,
                       prefix: str = "",
                       attn_type: str = AttentionType.DECODER,
                       params_dtype: Optional[mstype.Type] = mstype.bfloat16,
                       **extra_weight_attrs):
        """
        for an attention layer.
        """
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

        self.use_pfa = use_pfa
        self.flatten = not self.use_pfa
        self.block_size = cache_config.block_size

        self.scale = float(scale)
        self.sparse_mode = 0 if is_310p() else 2
        pre_tokens = 2147483647
        next_tokens = 2147483647

        self.reshape_and_cache = ReshapeAndCache()
        self.reshape = ops.auto_generate.ReshapeView()
        self.use_fused_attn = self.use_pfa and not is_310p()
        if self.use_fused_attn:
            self.fused_attention = ops.auto_generate.FusedInferAttentionScore(
                num_heads=num_heads,
                scale_value=self.scale,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
                input_layout="BSH",
                num_key_value_heads=num_kv_heads,
                sparse_mode=self.sparse_mode,
                inner_precise=1,
                block_size=self.block_size)
        else:
            if self.use_pfa:
                self.flash_attention = PromptFlashAttention(
                    num_heads=num_heads,
                    scale_value=self.scale,
                    pre_tokens=pre_tokens,
                    next_tokens=next_tokens,
                    input_layout="BSH",
                    num_key_value_heads=num_kv_heads,
                    sparse_mode=self.sparse_mode,
                    inner_precise=1)
            else:
                self.flash_attention = FlashAttentionScore(
                    head_num=num_heads,
                    scale_value=self.scale,
                    pre_tokens=pre_tokens,
                    next_tokens=next_tokens,
                    input_layout="TH")

            self.paged_attention = PagedAttention(head_num=num_heads,
                                                  scale_value=self.scale,
                                                  kv_head_num=num_kv_heads)

    def apply(
        self,
        layer: mindspore.nn.Cell,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        slot_mapping: Tensor,
        attn_mask: Tensor,
        batch_valid_length: Tensor,
        q_seq_lens: Tensor,
        block_tables: Tensor,
    ) -> Tensor:
        output = query
        key = key.contiguous()
        value = value.contiguous()
        cache_out = self.reshape_and_cache(key, value, key_cache, value_cache,
                                           slot_mapping)
        query = ops.depend(query, cache_out)
        is_prefill = get_model_context("is_prefill")
        if self.use_fused_attn:
            if not is_prefill:
                key = self.reshape(
                    key_cache,
                    (-1, self.block_size, self.kv_hidden_size_per_partition))
                value = self.reshape(
                    value_cache,
                    (-1, self.block_size, self.kv_hidden_size_per_partition))
                attn_mask = None
            actual_seq_lengths_kv = batch_valid_length if not is_prefill \
                                    else None
            block_tables = block_tables if not is_prefill else None
            output = self.fused_attention(
                query, [key], [value],
                None,
                attn_mask,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                block_table=block_tables)[0]
        elif is_prefill:
            output = self._run_prefill_forward(query, key, value, attn_mask,
                                               batch_valid_length,
                                               batch_valid_length)
        else:
            output = self._run_decode_forward(layer, query, key_cache,
                                              value_cache, block_tables,
                                              batch_valid_length, attn_mask,
                                              q_seq_lens)
        return output

    def _run_prefill_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor,
        actual_seq_qlen: tuple[int],
        actual_seq_kvlen: tuple[int],
    ) -> Tensor:
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
        if self.use_pfa:
            output = self.flash_attention(query, key, value, attn_mask)
        else:
            query = query.view(-1, self.hidden_size_per_partition)
            key = key.view(-1, self.kv_hidden_size_per_partition)
            value = value.view(-1, self.kv_hidden_size_per_partition)
            _, _, _, output = self.flash_attention(query, key, value, None,
                                                   None, None, attn_mask, None,
                                                   actual_seq_qlen,
                                                   actual_seq_kvlen)
            output = output.view(1, -1, self.hidden_size_per_partition)
        return output

    def _run_decode_forward(
        self,
        layer,
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
            key_cache: shape = [num_block, block_size, kv_heads_per_partition,
                                head_size]
            value_cache: shape = [num_block, block_size, kv_heads_per_partition,
                                  head_size]
            block_tables: shape = [block_size, num_block]
            context_lens: shape = [batch_size, ]
        """
        output = self.paged_attention(query, key_cache, value_cache,
                                      block_tables, batch_valid_length, None,
                                      None, attn_mask, q_seq_lens)

        return output

    def process_weights_after_loading(self, layer: mindspore.nn.Cell) -> None:
        pass


class KVCacheInt8Method(BaseKVCacheMethod):
    """
    Quant method that adds `_k_scale` and `_v_scale` attributes to the
    Attention layer to support loading those scaling factors from checkpoints.
    The k/v_scale will be used to:
        - quantize k/v_cache entries before saving them to the cache
        - dequantize k/v_cache entries before fetching them from the cache

    :param quant_config: the appropriate QuantizationConfig
    """

    def __init__(self, quant_config: QuantizationConfig = None):
        self.quant_config = quant_config
        self.quant = QuantV2()

    def create_weights(self,
                       layer: mindspore.nn.Cell,
                       num_heads: int,
                       head_size: int,
                       scale: float,
                       num_kv_heads: Optional[int] = None,
                       use_pfa: bool = False,
                       cache_config: Optional[CacheConfig] = None,
                       prefix: str = "",
                       attn_type: str = AttentionType.DECODER,
                       params_dtype: Optional[mstype.Type] = mstype.bfloat16,
                       **extra_weight_attrs):
        """
        Create "weight" (aka k_scale and v_scale)
        for an attention layer.
        """
        if not num_kv_heads:
            num_kv_heads = num_heads
        self.attn_type = attn_type
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.hidden_size_per_partition = num_heads * head_size
        self.kv_hidden_size_per_partition = num_kv_heads * head_size
        self.flatten = True
        # pynative 下不支持拉平操作。
        input_layout = "TH" if self.flatten else "BSH"
        self.scale = float(scale)
        pre_tokens = 2147483647
        next_tokens = 2147483647
        self.reshape_and_cache = ReshapeAndCache()
        self.flash_attention = FlashAttentionScore(head_num=num_heads,
                                                   scale_value=self.scale,
                                                   pre_tokens=pre_tokens,
                                                   next_tokens=next_tokens,
                                                   input_layout=input_layout)
        self.paged_attention = PagedAttention(head_num=num_heads,
                                              scale_value=self.scale,
                                              kv_head_num=num_kv_heads)
        kv_scale = Parameter(initializer(
            'zeros', (2, self.kv_hidden_size_per_partition), dtype.float16),
                             name="kv_scale",
                             requires_grad=False)
        kv_offset = Parameter(initializer(
            'zeros', (2, self.kv_hidden_size_per_partition), dtype.float16),
                              name="kv_offset",
                              requires_grad=False)
        k_scale = Parameter(initializer('zeros',
                                        (self.kv_hidden_size_per_partition),
                                        params_dtype),
                            name="k_scale",
                            requires_grad=False)
        v_scale = Parameter(initializer('zeros',
                                        (self.kv_hidden_size_per_partition),
                                        params_dtype),
                            name="v_scale",
                            requires_grad=False)
        k_offset = Parameter(initializer('zeros',
                                         (self.kv_hidden_size_per_partition),
                                         params_dtype),
                             name="k_offset",
                             requires_grad=False)
        v_offset = Parameter(initializer('zeros',
                                         (self.kv_hidden_size_per_partition),
                                         params_dtype),
                             name="v_offset",
                             requires_grad=False)
        set_weight_attrs(kv_scale, {"output_dim": 1})
        set_weight_attrs(kv_offset, {"output_dim": 1})
        set_weight_attrs(k_scale, {"output_dim": 0})
        set_weight_attrs(v_scale, {"output_dim": 0})
        set_weight_attrs(k_offset, {"output_dim": 0})
        set_weight_attrs(v_offset, {"output_dim": 0})
        set_weight_attrs(kv_scale, extra_weight_attrs)
        set_weight_attrs(kv_offset, extra_weight_attrs)
        set_weight_attrs(k_scale, extra_weight_attrs)
        set_weight_attrs(v_scale, extra_weight_attrs)
        set_weight_attrs(k_offset, extra_weight_attrs)
        set_weight_attrs(v_offset, extra_weight_attrs)

        layer.insert_param_to_cell("kv_scale", kv_scale)
        layer.insert_param_to_cell("kv_offset", kv_offset)
        layer.insert_param_to_cell("k_scale", k_scale)
        layer.insert_param_to_cell("v_scale", v_scale)
        layer.insert_param_to_cell("k_offset", k_offset)
        layer.insert_param_to_cell("v_offset", v_offset)

    def apply(
        self,
        layer: mindspore.nn.Cell,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        slot_mapping: Tensor,
        attn_mask: Tensor,
        batch_valid_length: Tensor,
        q_seq_lens: Tensor,
        block_tables: Tensor,
    ) -> Tensor:
        output = query
        k_scale = layer.k_scale
        v_scale = layer.v_scale
        k_offset = layer.k_offset
        v_offset = layer.v_offset
        quant_key = self.quant(key, k_scale, k_offset, False, "ROUND",
                               dtype.int8)
        quant_value = self.quant(value, v_scale, v_offset, False, "ROUND",
                                 dtype.int8)
        cache_out = self.reshape_and_cache(quant_key, quant_value, key_cache,
                                           value_cache, slot_mapping)
        query = ops.depend(query, cache_out)
        if get_model_context("is_prefill"):
            output = self._run_prefill_forward(query, key, value, attn_mask,
                                               batch_valid_length,
                                               batch_valid_length)
        else:
            output = self._run_decode_forward(layer, query, key_cache,
                                              value_cache, block_tables,
                                              batch_valid_length, attn_mask,
                                              q_seq_lens)
        return output

    def process_weights_after_loading(self, layer: mindspore.nn.Cell) -> None:
        k_scale = layer.k_scale.asnumpy()
        v_scale = layer.v_scale.asnumpy()
        k_offset = layer.k_offset.asnumpy()
        v_offset = layer.v_offset.asnumpy()
        layer.k_offset = Parameter(Tensor(k_offset, dtype=mindspore.int8),
                                   name=layer.k_offset.name,
                                   requires_grad=False)
        layer.v_offset = Parameter(Tensor(v_offset, dtype=mindspore.int8),
                                   name=layer.v_offset.name,
                                   requires_grad=False)
        ic = k_scale.shape[0]
        kv_scale = np.concatenate((k_scale.reshape(
            (1, ic)), v_scale.reshape((1, ic))))
        kv_offset = np.concatenate((k_offset.reshape(
            (1, ic)), v_offset.reshape((1, ic))))
        layer.kv_scale = Parameter(Tensor(kv_scale,
                                          dtype=layer.kv_scale.dtype),
                                   name=layer.kv_scale.name,
                                   requires_grad=False)
        layer.kv_offset = Parameter(Tensor(kv_offset,
                                           dtype=layer.kv_offset.dtype),
                                    name=layer.kv_offset.name,
                                    requires_grad=False)

    def _run_prefill_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor,
        actual_seq_qlen: tuple[int],
        actual_seq_kvlen: tuple[int],
    ) -> Tensor:
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
        query = query.view(-1, self.hidden_size_per_partition)
        key = key.view(-1, self.kv_hidden_size_per_partition)
        value = value.view(-1, self.kv_hidden_size_per_partition)
        _, _, _, output = self.flash_attention(query, key, value, None, None,
                                               None, attn_mask, None,
                                               actual_seq_qlen,
                                               actual_seq_kvlen)
        output = output.view(1, -1, self.hidden_size_per_partition)
        return output

    def _run_decode_forward(
        self,
        layer,
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
            key_cache: shape = [num_block, block_size,
                                kv_heads_per_partition, head_size]
            value_cache: shape = [num_block, block_size,
                                  kv_heads_per_partition, head_size]
            block_tables: shape = [block_size, num_block]
            context_lens: shape = [batch_size, ]
        """
        kv_scale = layer.kv_scale
        kv_offset = layer.kv_offset
        output = self.paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            batch_valid_length,
            kv_scale,
            kv_offset,
            attn_mask,
            q_seq_lens,
        )
        return output
