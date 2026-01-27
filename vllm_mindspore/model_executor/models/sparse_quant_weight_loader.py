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

from mindspore import Parameter, Tensor, ops
from vllm.distributed import get_tensor_model_parallel_rank

from vllm_mindspore.utils import FORMAT_TYPE, is_310p


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


def adjust_sparse_quant_weights_for_310p(params_dict: dict[str, Parameter],
                                         quant_config,
                                         target_keywords: list[str] = None):
    """Adjust sparse quantized weight format for 310P platform.
    
    Note: Compressed sparse weights should NOT be converted to Nz format,
    only regular weights need format conversion.
    
    Args:
        params_dict: Dictionary of parameter names to Parameter objects
        quant_config: Quantization configuration object
        target_keywords: List of weight name keywords to adjust.
                       Default: ["qkv_proj.weight", "o_proj.weight",
                                "gate_up_proj.weight", "down_proj.weight",
                                "lm_head.weight"]
    """
    if not is_310p():
        return

    if target_keywords is None:
        target_keywords = [
            "qkv_proj.weight",
            "o_proj.weight",
            "gate_up_proj.weight",
            "down_proj.weight",
            "lm_head.weight",
        ]

    for name, param in params_dict.items():
        if any(name.endswith(keyword) for keyword in target_keywords):
            # Check if this is a sparse quantized weight
            # Compressed sparse weights should NOT be converted to Nz format
            if is_sparse_quant_weight(name, quant_config):
                continue
            # Convert other weights to Nz format
            cast_weight = ops.auto_generate.format_cast(
                param, FORMAT_TYPE['nz'])
            param.set_data(cast_weight)


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
        # Skip quant_bias for non-zero ranks in tensor parallelism
        if (get_tensor_model_parallel_rank() > 0
                and "o_proj.quant_bias" in name):
            continue
        if name not in params_dict:
            continue
        param = params_dict[name]
        # Load full weight directly using [:] to avoid any slicing
        param.set_data(Tensor(loaded_weight[:]).contiguous())
        loaded_params.add(name)

        # Handle tie_word_embeddings
        tie_params = handle_tie_word_embeddings(name, param, params_dict,
                                                config)
        loaded_params.update(tie_params)

    # 310P platform format adjustment
    adjust_sparse_quant_weights_for_310p(params_dict, quant_config)

    return loaded_params
