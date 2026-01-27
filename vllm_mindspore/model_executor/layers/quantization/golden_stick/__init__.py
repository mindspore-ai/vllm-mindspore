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

# isort: off
from vllm_mindspore.model_executor.layers.quantization.golden_stick.\
    a8w8 import A8W8LinearMethod
from vllm_mindspore.model_executor.layers.quantization.golden_stick.\
    a8w8sc import A8W8SCLinearMethod
from vllm_mindspore.model_executor.layers.quantization.golden_stick.\
    golden_stick import GoldenStickConfig, ModelSlimConfig
from vllm_mindspore.model_executor.model_loader.sparse_quant_loader import (
    SparseQuantModelLoader)
# isort: on

# Note: A8W8SCLinearMethod (W8A8SC sparse quantization) is specific to
# the golden_stick quantization framework. It is configured through
# GoldenStickConfig and uses rank-level quantization configuration.
# This method is not a general-purpose quantization method and should
# remain in the golden_stick module.
__all__ = [
    "GoldenStickConfig",
    "ModelSlimConfig",
    "A8W8LinearMethod",
    "A8W8SCLinearMethod",
    "SparseQuantModelLoader",
]
