# type: ignore
# isort:skip_file
# SPDX-License-Identifier: Apache-2.0
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
"""test_attention"""
import mindspore
import numpy as np
import pytest
from mindspore import Parameter, Tensor, context, mint
from mindspore import dtype as mstype
from mindspore import nn

from vllm_mindspore.model_executor.layers.quantization.attention import (
    BaseKVCacheMethod, KVCacheInt8Method)
from vllm.config import CacheConfig
from vllm.attention.backends.abstract import AttentionType

from tests.utils.common_utils import teardown_function, setup_function


@pytest.fixture(params=[mstype.float16, mstype.bfloat16],
                ids=["float16", "bfloat16"])
def params_dtype(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["prefill", "decode"])
def is_prefill(request):
    return request.param


class SimpleAttention(nn.Cell):

    def __init__(self, method, params_dtype):
        super().__init__()
        self.method = method
        self.params_dtype = params_dtype

    def construct(self, query: Tensor, key: Tensor, value: Tensor,
                  key_cache: Tensor, value_cache: Tensor, is_prefill: bool,
                  slot_mapping: Tensor, attn_mask: Tensor,
                  batch_valid_length: Tensor, q_seq_lens: Tensor,
                  block_tables: Tensor):
        return self.method.apply(layer=self,
                                 query=query,
                                 key=key,
                                 value=value,
                                 key_cache=key_cache,
                                 value_cache=value_cache,
                                 is_prefill=is_prefill,
                                 slot_mapping=slot_mapping,
                                 attn_mask=attn_mask,
                                 batch_valid_length=batch_valid_length,
                                 q_seq_lens=q_seq_lens,
                                 block_tables=block_tables)


class TestBaseKVCacheMethod:

    def setup_base_kv_cache(self, params_dtype):
        """Setup base kv cache method."""
        context.set_context(jit_config={
            "jit_level": "O0",
            "infer_boost": "on"
        })
        context.set_context(mode=context.GRAPH_MODE)
        cache_config = CacheConfig(block_size=16,
                                   gpu_memory_utilization=-1,
                                   swap_space=-1,
                                   cache_dtype="auto")
        method = BaseKVCacheMethod()
        # create mock linear
        layer = SimpleAttention(method, params_dtype)

        method.create_weights(layer=layer,
                              num_heads=8,
                              head_size=128,
                              scale=0.5,
                              num_kv_heads=8,
                              use_pfa=False,
                              cache_config=cache_config,
                              attn_type=AttentionType.DECODER,
                              params_dtype=params_dtype)
        return method, layer

    def test_create_weights(self, params_dtype):
        """Test base kvCache create weight method."""
        method, _ = self.setup_base_kv_cache(params_dtype)

        # verify the correctness of the attr
        assert method.num_heads == 8
        assert method.num_kv_heads == 8
        assert method.head_size == 128
        assert method.scale == 0.5
        assert method.block_size == 16

    def test_apply(self, is_prefill, params_dtype):
        """Test base kvCache apply method."""
        _, layer = self.setup_base_kv_cache(params_dtype)

        # prepare to input data
        if is_prefill:
            seq_len = 6
            batch_valid_length = 6
            prefill_mask_coeff = 1.0 if params_dtype == mstype.bfloat16 \
                                else -10000.0
            attn_mask = Tensor(np.triu(np.ones(shape=(128, 128), \
                                               dtype=np.float16), k=1) * \
                               prefill_mask_coeff, dtype=params_dtype)
        else:
            seq_len = 1
            batch_valid_length = 1
            attn_mask = mint.zeros((1, 1), dtype=params_dtype)

        q_seq_lens_np = np.array(seq_len)
        slot_mapping = Tensor(np.arange(seq_len), mstype.int32)
        q_seq_lens = Tensor(q_seq_lens_np, mstype.int32)
        batch_valid_length = Tensor([batch_valid_length], mstype.int32)

        batch_size, head_num, head_dim = 1, 8, 128
        n_kv_head = 8
        hidden_size = head_num * head_dim
        kv_hidden_size = n_kv_head * head_dim

        query = Tensor(np.random.rand(batch_size, seq_len, hidden_size),
                       dtype=params_dtype)
        key = Tensor(np.random.rand(batch_size, seq_len, kv_hidden_size),
                     dtype=params_dtype)
        value = Tensor(np.random.rand(batch_size, seq_len, kv_hidden_size),
                       dtype=params_dtype)

        # prepare to kv cache
        num_blocks = 100
        block_size = 16
        key_cache = Tensor(np.zeros(
            (num_blocks, block_size, n_kv_head, head_dim)),
                           dtype=params_dtype)
        value_cache = Tensor(np.zeros(
            (num_blocks, block_size, n_kv_head, head_dim)),
                             dtype=params_dtype)
        block_tables = Tensor(
            np.zeros((batch_size, num_blocks)).astype(np.int32))

        # perform apply
        output = layer(query=query,
                       key=key,
                       value=value,
                       key_cache=key_cache,
                       value_cache=value_cache,
                       is_prefill=is_prefill,
                       slot_mapping=slot_mapping,
                       attn_mask=attn_mask,
                       batch_valid_length=batch_valid_length,
                       q_seq_lens=q_seq_lens,
                       block_tables=block_tables)

        # verify the output
        assert output.shape == (batch_size, seq_len, hidden_size)
        assert output.dtype == params_dtype


class TestKVCacheInt8Method:

    def setup_kv_cache_int8(self, params_dtype):
        """Setup kvCache int8 method."""
        context.set_context(jit_config={
            "jit_level": "O0",
            "infer_boost": "on"
        })
        context.set_context(mode=context.GRAPH_MODE)
        cache_config = None
        method = KVCacheInt8Method()

        layer = SimpleAttention(method, params_dtype)
        method.create_weights(layer=layer,
                              num_heads=8,
                              head_size=128,
                              scale=0.5,
                              num_kv_heads=8,
                              use_pfa=False,
                              cache_config=cache_config,
                              attn_type=AttentionType.DECODER,
                              params_dtype=params_dtype)
        return method, layer

    def test_create_weights(self, params_dtype):
        """Test kvCache int8 create weight method."""
        method, _ = self.setup_kv_cache_int8(params_dtype)

        # verify the correctness of the attr
        assert method.num_heads == 8
        assert method.num_kv_heads == 8
        assert method.head_size == 128
        assert method.scale == 0.5

    def test_apply(self, is_prefill, params_dtype):
        """Test kvCache int8 apply method."""
        _, layer = self.setup_kv_cache_int8(params_dtype)

        # prepare to input data
        if is_prefill:
            seq_len = 6
            batch_valid_length = 6
            prefill_mask_coeff = 1.0 if params_dtype == mstype.bfloat16 \
                                else -10000.0
            attn_mask = Tensor(np.triu(np.ones(shape=(128, 128), \
                                               dtype=np.float16), k=1) * \
                               prefill_mask_coeff, dtype=params_dtype)
        else:
            seq_len = 1
            batch_valid_length = 1
            attn_mask = mint.zeros((1, 1), dtype=params_dtype)

        q_seq_lens_np = np.array(seq_len)
        slot_mapping = Tensor(np.arange(seq_len), mstype.int32)
        q_seq_lens = Tensor(q_seq_lens_np, mstype.int32)
        batch_valid_length = Tensor([batch_valid_length], mstype.int32)

        batch_size, head_num, head_dim = 1, 8, 128
        n_kv_head = 8
        hidden_size = head_num * head_dim
        kv_hidden_size = n_kv_head * head_dim

        query = Tensor(np.random.rand(batch_size, seq_len, hidden_size),
                       dtype=params_dtype)
        key = Tensor(np.random.rand(batch_size, seq_len, kv_hidden_size),
                     dtype=params_dtype)
        value = Tensor(np.random.rand(batch_size, seq_len, kv_hidden_size),
                       dtype=params_dtype)

        # prepare to kv cache
        num_blocks = 100
        block_size = 16
        key_cache = Tensor(np.zeros(
            (num_blocks, block_size, n_kv_head, head_dim)),
                           dtype=mindspore.int8)
        value_cache = Tensor(np.zeros(
            (num_blocks, block_size, n_kv_head, head_dim)),
                             dtype=mindspore.int8)
        block_tables = Tensor(
            np.zeros((batch_size, num_blocks)).astype(np.int32))

        # perform apply
        output = layer(query=query,
                       key=key,
                       value=value,
                       key_cache=key_cache,
                       value_cache=value_cache,
                       is_prefill=is_prefill,
                       slot_mapping=slot_mapping,
                       attn_mask=attn_mask,
                       batch_valid_length=batch_valid_length,
                       q_seq_lens=q_seq_lens,
                       block_tables=block_tables)

        # verify output
        assert output.shape == (batch_size, seq_len, hidden_size)
        assert output.dtype == params_dtype

    def test_process_weights_after_loading(self, params_dtype):
        """Test kvCache int8 process weights after loading."""
        method, layer = self.setup_kv_cache_int8(params_dtype)

        # set the initial weight value
        kv_hidden_size = 8 * 128
        k_scale_val = np.random.rand(kv_hidden_size).astype(np.float32)
        v_scale_val = np.random.rand(kv_hidden_size).astype(np.float32)
        k_offset_val = np.random.rand(kv_hidden_size).astype(np.float32)
        v_offset_val = np.random.rand(kv_hidden_size).astype(np.float32)

        layer.k_scale = Parameter(Tensor(k_scale_val),
                                  name="k_scale",
                                  requires_grad=False)
        layer.v_scale = Parameter(Tensor(v_scale_val),
                                  name="v_scale",
                                  requires_grad=False)
        layer.k_offset = Parameter(Tensor(k_offset_val),
                                   name="k_offset",
                                   requires_grad=False)
        layer.v_offset = Parameter(Tensor(v_offset_val),
                                   name="v_offset",
                                   requires_grad=False)

        # process weights
        method.process_weights_after_loading(layer)

        # verify shape and dtype
        assert layer.k_offset.dtype == mindspore.int8
        assert layer.v_offset.dtype == mindspore.int8

        assert layer.kv_scale.shape == (2, kv_hidden_size)
        assert layer.kv_offset.shape == (2, kv_hidden_size)

        assert layer.kv_scale.dtype == mindspore.float16
        assert layer.kv_offset.dtype == mindspore.float16


@pytest.mark.skip(
    reason="open this after ReshapeView ops merge into mindspore.")
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_base_kvcache_method(is_prefill, params_dtype):
    base_kvcache_method = TestBaseKVCacheMethod()
    base_kvcache_method.test_create_weights(params_dtype)
    base_kvcache_method.test_apply(is_prefill, params_dtype)


@pytest.mark.skip(
    reason="open this after ReshapeView ops merge into mindspore.")
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_int8_kvcache_method(is_prefill, params_dtype):
    int8_kvcache_method = TestKVCacheInt8Method()
    int8_kvcache_method.test_create_weights(params_dtype)
    int8_kvcache_method.test_apply(is_prefill, params_dtype)
    int8_kvcache_method.test_process_weights_after_loading(params_dtype)
