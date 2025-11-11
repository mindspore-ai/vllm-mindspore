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

import re

from mindspore.common import dtype as mstype
from mindspore import nn
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)

from vllm_mindspore.model_executor.layers.quantization.golden_stick.a8w8 \
    import A8W8LinearMethod
from vllm_mindspore.model_executor.layers.linear import (UnquantizedLinearMethod
                                                        )
from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)


class GoldenStickConfig(QuantizationConfig):
    concat_linear_mapping = {
        "q_proj": "qkv_proj",
        "gate_proj": "gate_up_proj",
    }

    quantization_method_mapping = {
        "FLOAT": UnquantizedLinearMethod,
        "W8A8": A8W8LinearMethod,
    }

    def __init__(self, config: dict[str, any]) -> None:
        super().__init__()
        self.is_modelslim = False
        self.config = config

    def get_name(self) -> str:
        return "golden_stick"

    def get_supported_act_dtypes(self) -> list["mstype"]:
        return [mstype.bfloat16, mstype.int8]

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_description.json",
                "quant_model_description.json"]

    @classmethod
    def from_config(cls, config: dict[str, any]) -> "QuantizationConfig":
        return cls(config)

    def get_quant_method(self, layer: nn.Cell,
                         prefix: str) -> QuantizeMethodBase:
        if isinstance(layer, VocabParallelEmbedding):
            return None
        quant_strategy = None

        for key, value in self.config.items():
            for uncat_name, cat_name in self.concat_linear_mapping.items():
                key = key.replace(uncat_name, cat_name)
            if "experts." in prefix:
                key = re.sub(r"experts\.\d+", "exports", key)
            if prefix in key:
                quant_strategy = value
                break

        if not quant_strategy:
            raise ValueError(
                f"Cannot find quantization strategy for layer {prefix}.")
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
