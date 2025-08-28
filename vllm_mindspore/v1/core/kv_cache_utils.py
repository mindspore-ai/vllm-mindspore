# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/v1/core/kv_cache_utils.py
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

from vllm.config import VllmConfig
from vllm.utils import cdiv
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_config_uniform_page_size, _get_kv_cache_config_uniform_type,
    check_enough_kv_cache_memory, create_kv_cache_group_specs,
    is_kv_cache_page_size_uniform, is_kv_cache_type_uniform, logger,
    unify_hybrid_kv_cache_specs)
from vllm.v1.kv_cache_interface import (KVCacheConfig, KVCacheSpec,
                                        KVCacheTensor)


def get_max_concurrency_for_kv_cache_config_diff_page_size(
        vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> float:
    """
    Get the maximum concurrency for the given KV cache configuration.
    """
    block_size = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
    num_block_per_request = cdiv(vllm_config.model_config.max_model_len,
                                 block_size)
    max_concurrency = kv_cache_config.num_blocks / num_block_per_request
    return max_concurrency


def _get_kv_cache_config_not_uniform(vllm_config: VllmConfig,
                                     kv_cache_spec: dict[str, KVCacheSpec],
                                     available_memory: int) -> KVCacheConfig:
    """
    Generates the KV cache configuration for a model with different page_size
    of KV cache. now only use for fa3 quant deepseek network, in the case have
    two AttentionSpec:
        MLAQuantFullAttentionSpec for fa3 quant layer
        FullAttentionSpec for not fa3 quant layer
    Divide the available memory equally among all layers.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The generated KVCacheConfig
    """

    # different layers may have different page_size_bytes
    page_sizes = sum(
        [layer.page_size_bytes for layer in kv_cache_spec.values()])
    num_blocks = int(available_memory // page_sizes)
    # create different groups based on page_size_bytes
    page_size_layer_map: dict[int, list[str]] = {}
    kv_cache_tensors = []

    for layer_name, layer in kv_cache_spec.items():
        kv_cache_tensors.append(
            KVCacheTensor(size=num_blocks * layer.page_size_bytes,
                          shared_by=[layer_name]))
        page_size_layer_map.setdefault(layer.page_size_bytes,
                                       []).append(layer_name)

    grouped_layer_names = list(page_size_layer_map.values())
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=create_kv_cache_group_specs(kv_cache_spec,
                                                    grouped_layer_names),
    )

    num_tokens = num_blocks * vllm_config.cache_config.block_size
    num_tokens_str = f"{num_tokens:,}"
    logger.info("NPU KV cache size: %s tokens", num_tokens_str)
    max_model_len_str = f"{vllm_config.model_config.max_model_len:,}"
    max_concurrency = get_max_concurrency_for_kv_cache_config_diff_page_size(
        vllm_config, kv_cache_config)
    logger.info("Maximum concurrency for %s tokens per request: %.2fx",
                max_model_len_str, max_concurrency)
    return kv_cache_config


# Compared to the native vLLM, the _get_kv_cache_config_not_uniform method
# is added to support DeepSeek FA3 quant. Since some layers in fa3 quant
# DeepSeek network are not quant layers with non-uniform page sizes,
# and the native vLLM does not support varying page sizes across layers,
# this new method is implemented to handle such cases.
def get_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
) -> KVCacheConfig:
    """
    Generates the KV cache configuration for a model.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The generated KVCacheConfigs
    """
    check_enough_kv_cache_memory(vllm_config, kv_cache_spec, available_memory)

    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)

    if is_kv_cache_type_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for
        # most models. Allocate the same amount of memory for
        # each layer.
        return _get_kv_cache_config_uniform_type(vllm_config, kv_cache_spec,
                                                 available_memory)
    elif is_kv_cache_page_size_uniform(kv_cache_spec):
        # Model contains multiple attention types, but KV cache of all layers
        # have the same physical memory per block per layer. Split the layers
        # into groups with the same number of layers, and thus same total page
        # size.
        return _get_kv_cache_config_uniform_page_size(vllm_config,
                                                      kv_cache_spec,
                                                      available_memory)

    return _get_kv_cache_config_not_uniform(vllm_config, kv_cache_spec,
                                            available_memory)
