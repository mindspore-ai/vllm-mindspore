# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2025-2026 Huawei Technologies Co., Ltd.
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
"""Utility functions for loading sparse quantized weights.

This module provides common functionality for loading sparse quantized
weights (W8A8SC) that can be shared across different model implementations
(e.g., Qwen2, Llama, Qwen3, etc.).
"""

from collections.abc import Iterable

import mindspore as ms
from mindspore import Parameter, Tensor

from vllm_mindspore.utils import cast_weight_for_310p, is_310p


def is_sparse_quant_weight(name: str, quant_config) -> bool:
    """Check if a weight is sparse quantized based on quantization config.
    
    Args:
        name: Weight name (e.g., "model.layers.0.self_attn.qkv_proj.weight")
        quant_config: Quantization configuration object with config attribute
    
    Returns:
        True if the weight is sparse quantized (W8A8S), False otherwise
    """
    if quant_config is not None and hasattr(quant_config, 'config'):
        from mindspore.communication import get_rank
        rank_id = get_rank()
        rank_key = f"rank_{rank_id}"
        if rank_key in quant_config.config:
            weight_type = quant_config.config[rank_key].get(name, "")
            if isinstance(weight_type, str) and weight_type.lower() == "w8a8s":
                return True
    return False


def handle_tie_word_embeddings(name: str, param: Parameter,
                               params_dict: dict[str, Parameter],
                               config) -> set[str]:
    """Handle tie_word_embeddings by sharing weights between embed_tokens
    and lm_head.
    
    When loading embed_tokens, also set lm_head.weight to share the same
    Tensor object to maintain weight sharing.
    
    Args:
        name: Current weight name being loaded
        param: Parameter object for the current weight
        params_dict: Dictionary of all parameter names to Parameter objects
        config: Model configuration object with tie_word_embeddings attribute
    
    Returns:
        Set of parameter names that were loaded (including "lm_head.weight"
        if tie_word_embeddings is enabled)
    """
    loaded_params: set[str] = set()
    if (config.tie_word_embeddings and "embed_tokens" in name
            and "lm_head.weight" in params_dict):
        lm_head_param = params_dict["lm_head.weight"]
        # Share the same Tensor object to maintain weight sharing
        lm_head_param.set_data(param.data)
        loaded_params.add("lm_head.weight")
    return loaded_params


def load_split_weights(weights: Iterable[tuple[str, Tensor]],
                       params_dict: dict[str, Parameter], config,
                       quant_config) -> set[str]:
    """Load sparse quantized weights directly without sharding.
    
    Weights are already partitioned by rank folders, so load them
    directly without any sharding operations. This function handles:
    1. Loading weights from rank-partitioned folders
    2. Handling tie_word_embeddings for shared weights
    3. Adjusting weights for 310P platform if needed
    
    Args:
        weights: Iterable of (name, weight) tuples
        params_dict: Dictionary of parameter names to Parameter objects
        config: Model configuration object
        quant_config: Quantization configuration object
    
    Returns:
        Set of loaded parameter names
    """
    weights_dict = dict(weights)
    loaded_params: set[str] = set()
    for name, loaded_weight in weights_dict.items():
        if name not in params_dict:
            continue
        param = params_dict[name]
        # Weights are already partitioned by rank folders
        # loaded_weight[:] returns numpy array from safetensors (framework="np")
        weight_np = loaded_weight[:]
        # Only do dtype conversion for 310P platform
        # This avoids unnecessary memory copies for non-310P platforms
        if is_310p():
            weight_np = cast_weight_for_310p(weight_np)
        # Use from_numpy for better performance
        # since numpy arrays are already contiguous
        weight_data = ms.from_numpy(weight_np)
        param.set_data(weight_data)
        loaded_params.add(name)

        # Handle tie_word_embeddings
        tie_params = handle_tie_word_embeddings(name, param, params_dict,
                                                config)
        loaded_params.update(tie_params)

    return loaded_params
