#
# Copyright 2026 Huawei Technologies Co., Ltd
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
#

"""Tests for ATB paged_attention operator front-end integration."""

import random

import numpy as np
import pytest
import torch
import torch_npu

from mrt.torch.fx_backend import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual

# Paged attention test config: seq length, blocks, heads, head dims
MAX_SEQ_LEN = 1024
NUM_BLOCKS = 64
NUM_TOKENS = 2
BLOCK_SIZE = 128
KV_HEADS = 16
HEAD_SIZE = 288
NUM_HEADS = 32
HEAD_SIZE_V = 96


def run_paged_attention(query, key_cache, value_cache, block_table, context_lens,
                        num_kv_heads, num_heads, out):
    """Call _npu_paged_attention without workspace; operator allocates internally."""
    scale_value = 1.0 / (HEAD_SIZE ** 0.5)  # Python float constant for graph tracing
    torch_npu._npu_paged_attention(  # pylint: disable=protected-access
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        num_heads,
        scale_value,
        block_table,
        context_lens,
        out,
    )
    return out


def prepare_inputs(dtype):
    """Build query, KV cache, block tables and context lengths with fixed seed."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    query = np.random.uniform(-1, 1, (NUM_TOKENS, NUM_HEADS, HEAD_SIZE)).astype(dtype)
    key_cache = np.random.uniform(-1, 1, (NUM_BLOCKS, BLOCK_SIZE, KV_HEADS, HEAD_SIZE)).astype(dtype)
    value_cache = np.random.uniform(-1, 1, (NUM_BLOCKS, BLOCK_SIZE, KV_HEADS, HEAD_SIZE_V)).astype(dtype)

    context_lens = np.full(NUM_TOKENS, MAX_SEQ_LEN, dtype=np.int32)
    max_blocks_per_seq = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = np.array([
        [random.randint(0, NUM_BLOCKS - 1) for _ in range(max_blocks_per_seq)]
        for _ in range(NUM_TOKENS)
    ], dtype=np.int32)

    return query, key_cache, value_cache, block_tables, context_lens


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", (torch.float16,))
def test_atb_paged_attention_basic(dtype):
    """
    Feature: Test ATB paged_attention operator front-end API.
    Description: Eager and compiled graph results of _npu_paged_attention must match.
    Expectation: The result is correct within tolerance.
    """
    compiled_paged_attention = torch.compile(run_paged_attention, backend=fx_backend)

    query_np, key_cache_np, value_cache_np, block_tables_np, context_lens_np = prepare_inputs(
        np.float16 if dtype == torch.float16 else np.float32
    )

    query = torch.from_numpy(query_np).npu()
    key_cache = torch.from_numpy(key_cache_np).npu()
    value_cache = torch.from_numpy(value_cache_np).npu()
    block_table = torch.from_numpy(block_tables_np).npu()
    context_lens = torch.from_numpy(context_lens_np)  # CPU tensor

    out_eager = torch.zeros(NUM_TOKENS, NUM_HEADS, HEAD_SIZE_V, dtype=dtype).npu()

    eager_out = run_paged_attention(
        query, key_cache, value_cache, block_table, context_lens,
        KV_HEADS, NUM_HEADS, out_eager,
    )

    out_compiled = torch.zeros_like(out_eager)
    compiled_out = compiled_paged_attention(
        query, key_cache, value_cache, block_table, context_lens,
        KV_HEADS, NUM_HEADS, out_compiled,
    )

    AssertRtolEqual(compiled_out, eager_out)
