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

"""Tests for ATB reshape_and_cache operator front-end integration."""

import random

import numpy as np
import pytest
import torch
import torch_npu

from ms_inferrt.torch.fx_backend import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual

# Reshape-and-cache test config: tokens, blocks, heads, head dims
NUM_TOKENS = 8
NUM_HEADS = 2
HEAD_SIZE_K = 16
HEAD_SIZE_V = 8
BLOCK_SIZE = 16
NUM_BLOCKS = 32


def run_reshape_and_cache(key, value, key_cache, value_cache, slot_indices):
    """Call ATB _npu_reshape_and_cache; operator updates caches in-place."""
    torch_npu._npu_reshape_and_cache(  # pylint: disable=protected-access
        key,
        value,
        key_cache,
        value_cache,
        slot_indices,
    )
    return key_cache, value_cache


def prepare_inputs(dtype):
    """Build key, value, KV cache and slot indices with fixed seed."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    key = np.random.uniform(-1, 1, (NUM_TOKENS, NUM_HEADS, HEAD_SIZE_K)).astype(dtype)
    value = np.random.uniform(-1, 1, (NUM_TOKENS, NUM_HEADS, HEAD_SIZE_V)).astype(dtype)

    key_cache = np.random.uniform(
        -1, 1, (NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE_K)
    ).astype(dtype)
    value_cache = np.random.uniform(
        -1, 1, (NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE_V)
    ).astype(dtype)

    num_slots = BLOCK_SIZE * NUM_BLOCKS
    slot_list = random.sample(range(num_slots), NUM_TOKENS)
    slot_indices = np.array(slot_list, dtype=np.int32)

    return key, value, key_cache, value_cache, slot_indices


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", (torch.float16,))
def test_atb_reshape_and_cache_basic(dtype):
    """
    Feature: Test ATB reshape_and_cache operator front-end API.
    Description: Eager and compiled graph results of _npu_reshape_and_cache must match.
    Expectation: The result is correct within tolerance.
    """
    compiled_reshape_and_cache = torch.compile(run_reshape_and_cache, backend=fx_backend)

    key_np, value_np, key_cache_np, value_cache_np, slot_indices_np = prepare_inputs(
        np.float16 if dtype == torch.float16 else np.float32
    )

    key = torch.from_numpy(key_np).to(dtype=dtype).npu()
    value = torch.from_numpy(value_np).to(dtype=dtype).npu()
    key_cache_init = torch.from_numpy(key_cache_np).to(dtype=dtype).npu()
    value_cache_init = torch.from_numpy(value_cache_np).to(dtype=dtype).npu()
    slot_indices = torch.from_numpy(slot_indices_np).to(torch.int32).npu()

    # Eager run
    key_cache_eager = key_cache_init.clone()
    value_cache_eager = value_cache_init.clone()
    eager_key_cache, eager_value_cache = run_reshape_and_cache(
        key, value, key_cache_eager, value_cache_eager, slot_indices
    )

    # Compiled run
    key_cache_compiled = key_cache_init.clone()
    value_cache_compiled = value_cache_init.clone()
    compiled_key_cache, compiled_value_cache = compiled_reshape_and_cache(
        key, value, key_cache_compiled, value_cache_compiled, slot_indices
    )

    AssertRtolEqual(compiled_key_cache, eager_key_cache)
    AssertRtolEqual(compiled_value_cache, eager_value_cache)
