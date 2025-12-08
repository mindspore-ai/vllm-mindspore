# SPDX-License-Identifier: Apache-2.0

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

import regex as re
from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.communication import get_rank
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)

from vllm_mindspore.model_executor.layers.linear import (
    LinearBase, UnquantizedLinearMethod)

# isort: off
from vllm_mindspore.model_executor.layers.quantization.golden_stick.\
    a8w8 import A8W8LinearMethod
from vllm_mindspore.model_executor.layers.quantization.golden_stick.\
    a8w8sc import A8W8SCLinearMethod
# isort: on
from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm_mindspore.utils import is_310p

WEIGHT_PARTS = 2


def _validate_quantization_consistency(config: dict[str, str]):
    """
    Validate that the quantization configuration is consistent for the
    specified prefix. This checks that all weights under the prefix
    have the same quantization type and handles fused linear layers.
    Raises ValueError if there are conflicts in quantization types.

    Args:
        config (dict[str, str]): A dictionary mapping weight names to their
                                 quantization types.
    Example:
        config = {
            ...
            "model.layers.0.self_attn.q_proj.weight": "W8A8",
            "model.layers.0.self_attn.q_proj.smooth_scale": "W8A8",
            "model.layers.0.self_attn.k_proj.weight": "W8A8",
            "model.layers.0.self_attn.k_proj.smooth_scale": "W8A8",
            "model.layers.0.self_attn.v_proj.weight": "W8A8",
            "model.layers.0.self_attn.v_proj.smooth_scale": "W8A8",
            ...
        }
        The quantization types for q_proj.weight and q_proj.smooth_scale
        should be consistent, and similarly for k_proj and v_proj.
        Also, the quantization types for q_proj, k_proj, and v_proj
        should be consistent due to fusion into qkv_proj.
    """

    class ModuleQuantizationDescription:

        def __init__(self, module_name: str):
            self.module_name = module_name
            self.quant_type = None

        def register_weight_quant_type(self, quant_type: str):
            if self.quant_type is None:
                self.quant_type = quant_type  # type: ignore[assignment]
                return True
            return self.quant_type == quant_type

    class FusedQuantizationDescription(ModuleQuantizationDescription):
        fused_linear_mapping = {
            "q_proj": "qkv_proj",
            "k_proj": "qkv_proj",
            "v_proj": "qkv_proj",
            "gate_proj": "gate_up_proj",
            "up_proj": "gate_up_proj",
        }

    module_quant_descriptions: dict[str, ModuleQuantizationDescription] = {}
    fused_quant_descriptions: dict[str, FusedQuantizationDescription] = {}
    conflict_info = {}
    for weight_full_name, quant_type in config.items():
        parts = weight_full_name.split('.')
        # make sure it's a weight
        if len(parts) < WEIGHT_PARTS:
            continue
        # format with
        # model.layers.{layer_number}.{module}.{linear}.{weight} or
        # model.visual.blocks.{layer_number}.{module}.{linear}.{weight}
        module_name = '.'.join(parts[:-1])
        if module_name not in module_quant_descriptions:
            module_quant_descriptions[
                module_name] = ModuleQuantizationDescription(module_name)
        if not module_quant_descriptions[
                module_name].register_weight_quant_type(quant_type):
            conflict_info[module_name] = "inner_conflict"
        linear_name = parts[-WEIGHT_PARTS]
        if linear_name in FusedQuantizationDescription.fused_linear_mapping:
            fused_linear_name = \
                FusedQuantizationDescription.fused_linear_mapping[
                    linear_name]
            fused_module_name = module_name.replace(linear_name,
                                                    fused_linear_name)
            if fused_module_name not in fused_quant_descriptions:
                fused_quant_descriptions[
                    fused_module_name] = FusedQuantizationDescription(
                        module_name)
            if not fused_quant_descriptions[
                    fused_module_name].register_weight_quant_type(quant_type):
                conflict_info[fused_module_name] = "fused_conflict"
    if conflict_info:
        raise ValueError(f"conflict_info: {conflict_info}")


class GoldenStickConfig(QuantizationConfig):
    concat_linear_mapping = {
        "q_proj": "qkv_proj",
        "gate_proj": "gate_up_proj",
    }

    quantization_method_mapping = {
        "FLOAT": UnquantizedLinearMethod,
        "W8A8": A8W8LinearMethod,
        "W8A8S": A8W8SCLinearMethod,
    }

    def __init__(self, config: dict[str, str]) -> None:
        super().__init__()
        self.is_modelslim = False
        self.config = config
        # Check if device_type is "310P" in the config
        # This is read from quant_model_description.json
        self.quant_device_type = config.get("device_type", "910")

    def get_name(self) -> str:
        return "golden_stick"

    def get_supported_act_dtypes(self) -> list["mstype"]:
        return [mstype.bfloat16, mstype.float16, mstype.int8]

    @staticmethod
    def get_config_filenames() -> list[str]:
        # quant_model_description_w8a8sc.json is for sparse quant (W8A8SC)
        return [
            "quantization_description.json", "quant_model_description.json",
            "quant_model_description_w8a8sc.json"
        ]

    @classmethod
    def from_config(cls, config: dict[str, str]) -> "QuantizationConfig":
        """Create a config class from the model's quantization config.
        
        Args:
            config: Dictionary containing quantization configuration.
            
        Returns:
            GoldenStickConfig instance with is_modelslim=False.
        """
        _validate_quantization_consistency(config)
        return cls(config)

    def get_quant_method(self, layer: nn.Cell,
                         prefix: str) -> QuantizeMethodBase:
        if isinstance(layer, VocabParallelEmbedding):
            return None

        # Pre-compute common values
        is_linear_base = isinstance(layer, LinearBase)
        weight_key = f"{prefix}.weight"

        # Handle rank-level sparse quantization config (W8A8SC)
        if is_linear_base:
            rank_id = get_rank()
            rank_key = f'rank_{rank_id}'
            rank_config = self.config.get(rank_key)

            if isinstance(rank_config, dict):
                quant_type = rank_config.get(weight_key)
                # Early check: if sparse quantization detected and not 310P
                if (isinstance(quant_type, str)
                        and quant_type.lower() == "w8a8s"):
                    if not is_310p():
                        raise RuntimeError(
                            "Sparse quantization (W8A8SC) is only "
                            "supported on 310P platform. Current "
                            "device is not 310P.")

                    # Get compressed weight and index shapes
                    weight_shape_key = f"{prefix}.weight.shape"
                    index_shape_key = f"{prefix}.index.shape"
                    compress_weight_size = rank_config.get(weight_shape_key)
                    compress_index_size = rank_config.get(index_shape_key)

                    if (compress_weight_size is not None
                            and compress_index_size is not None):
                        return A8W8SCLinearMethod(self, compress_weight_size,
                                                  compress_index_size)

        # Handle flat quantization config
        quant_strategy = None
        for key, value in self.config.items():
            if key.startswith('rank_'):
                continue

            matched_key = key
            for uncat_name, cat_name in self.concat_linear_mapping.items():
                matched_key = matched_key.replace(uncat_name, cat_name)
            if "experts." in prefix:
                matched_key = re.sub(r"experts\.\d+", "experts", matched_key)
            if prefix in matched_key:
                quant_strategy = value
                break

        if not quant_strategy:
            print(f"No quantization strategy matched for prefix '{prefix}', "
                  f"using default: FLOAT")
            quant_strategy = "FLOAT"
        if quant_strategy not in self.quantization_method_mapping:
            raise ValueError(
                f"Unsupported quantization strategy: {quant_strategy} "
                f"for layer {prefix}.")
        if quant_strategy == "FLOAT":
            return UnquantizedLinearMethod()
        return self.quantization_method_mapping[quant_strategy](self)

    @classmethod
    def get_min_capability(cls) -> int:
        return -1


class ModelSlimConfig(GoldenStickConfig):
    """ModelSlimConfig for ModelSlim quantization.
    
    This class extends GoldenStickConfig with is_modelslim=True.
    It uses AclnnQuantBatchMatMul and scalar quantization parameters.
    """

    def __init__(self, config: dict[str, str]) -> None:
        super().__init__(config)
        self.is_modelslim = True

    @classmethod
    def from_config(cls, config: dict[str, str]) -> "QuantizationConfig":
        """Create a config class from the model's quantization config.
        
        Args:
            config: Dictionary containing quantization configuration.
            
        Returns:
            ModelSlimConfig instance with is_modelslim=True.
        """
        _validate_quantization_consistency(config)
        return cls(config)
