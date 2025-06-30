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

from mindspore import Tensor, mint, nn, jit
from mindspore.common import dtype as mstype

from vllm.config import CacheConfig
from vllm.attention.backends.abstract import AttentionType

from vllm_mindspore.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm_mindspore.model_executor.layers.quantization.attention import BaseKVCacheMethod
from vllm.distributed import get_tensor_model_parallel_rank

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
        use_pfa: bool = False,
        params_dtype: Optional[mstype.Type] = mstype.bfloat16,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        self.cache_config = cache_config
        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = BaseKVCacheMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)

        self.quant_method.create_weights(
            self,
            num_heads,
            head_size,
            scale,
            num_kv_heads=num_kv_heads,
            use_pfa=use_pfa,
            cache_config=cache_config,
            prefix=prefix,
            attn_type=attn_type,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader
        )

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
        return self.quant_method.apply(self, query, key, value, key_cache, value_cache, is_prefill,
                                       slot_mapping, attn_mask, batch_valid_length, q_seq_lens, block_tables)

    def weight_loader(self, param, loaded_weight):
        tp_rank = get_tensor_model_parallel_rank()
        output_dim = getattr(param, "output_dim", None)

        if output_dim is not None:
            shard_size = param.shape[output_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size).contiguous()

        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param.shape == loaded_weight.shape
        param.set_data(loaded_weight)

