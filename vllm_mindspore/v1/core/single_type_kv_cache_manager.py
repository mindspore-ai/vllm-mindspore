# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/v1/core/single_type_kv_cache_manager.py
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

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager, SingleTypeKVCacheManager, SlidingWindowManager)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheSpec,
                                        SlidingWindowSpec)

from vllm_mindspore.v1.kv_cache_interface import MLAQuantFullAttentionSpec


@classmethod  # type: ignore[misc]
def find_longest_cache_hit(
    cls,
    block_hashes: list[BlockHash],
    max_length: int,
    kv_cache_group_ids: list[int],
    block_pool: BlockPool,
    kv_cache_spec: KVCacheSpec,
    use_eagle: bool,
) -> tuple[list[KVCacheBlock], ...]:
    assert isinstance(
        kv_cache_spec, (FullAttentionSpec, MLAQuantFullAttentionSpec)), (
            "FullAttentionManager can only be used for full attention "
            "or mla quant full attention groups")
    computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
        [] for _ in range(len(kv_cache_group_ids)))
    max_num_blocks = max_length // kv_cache_spec.block_size
    for i, block_hash in zip(range(max_num_blocks), block_hashes):
        # block_hashes is a chain of block hashes. If a block hash is not
        # in the cached_block_hash_to_id, the following block hashes are
        # not computed yet for sure.
        if cached_block := block_pool.get_cached_block(block_hash,
                                                       kv_cache_group_ids):
            for computed, cached in zip(computed_blocks, cached_block):
                computed.append(cached)
        else:
            break
    if use_eagle and computed_blocks[0]:
        for computed in computed_blocks:
            computed.pop()
    return computed_blocks


spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {
    FullAttentionSpec: FullAttentionManager,
    MLAQuantFullAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager,
}
