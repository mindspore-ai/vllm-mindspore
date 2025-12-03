# SPDX-License-Identifier: Apache-2.0

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

import regex as re
from mindspore import nn
from mindspore.common import dtype as mstype
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)

from vllm_mindspore.model_executor.layers.linear import UnquantizedLinearMethod

# isort: off
from vllm_mindspore.model_executor.layers.quantization.golden_stick.\
    a8w8 import A8W8LinearMethod
# isort: on
from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)


def _validate_quantization_consistency(config: dict[str, str]):
    """
    Validate that the quantization configuration is consistent for the
    specified prefix. This checks that all weights under the prefix
    have the same quantization type and handles fused linear layers.
    Raises ValueError if there are conflicts in quantization types.
    """

    class ModuleQuantizationDescription:

        def __init__(self, module_name: str):
            self.module_name = module_name
            self.quant_type = None

        def register_weight_quant_type(self, quant_type: str):
            if self.quant_type is None:
                self.quant_type = quant_type
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
        linear_name = parts[-2]
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
    }

    def __init__(self, config: dict[str, str]) -> None:
        super().__init__()
        self.is_modelslim = False
        self.config = config

    def get_name(self) -> str:
        return "golden_stick"

    def get_supported_act_dtypes(self) -> list["mstype"]:
        return [mstype.bfloat16, mstype.float16, mstype.int8]

    @staticmethod
    def get_config_filenames() -> list[str]:
        return [
            "quantization_description.json", "quant_model_description.json"
        ]

    @classmethod
    def from_config(cls, config: dict[str, str]) -> "QuantizationConfig":
        _validate_quantization_consistency(config)
        return cls(config)

    def get_quant_method(self, layer: nn.Cell,
                         prefix: str) -> QuantizeMethodBase:
        if isinstance(layer, VocabParallelEmbedding):
            return None
        quant_strategy = None

        # replace prefix for multimodal model
        prefix = prefix.replace("language_model.model", "model.language_model")
        for key, value in self.config.items():
            for uncat_name, cat_name in self.concat_linear_mapping.items():
                key = key.replace(uncat_name, cat_name)
            if "experts." in prefix:
                key = re.sub(r"experts\.\d+", "exports", key)
            if prefix in key:
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
