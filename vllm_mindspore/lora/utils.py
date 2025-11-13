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
"""Unified interface for LoRA layers in vllm-mindspore."""

from vllm.lora.layers import (ColumnParallelLinearWithShardedLoRA,
                              MergedColumnParallelLinearWithShardedLoRA,
                              MergedQKVParallelLinearWithShardedLoRA,
                              QKVParallelLinearWithShardedLoRA,
                              RowParallelLinearWithShardedLoRA)

# yapf conflicts with isort for this block
# yapf: disable  # noqa: ERA001
from vllm_mindspore.lora.layers import (BaseLayerWithLoRA,
                                        ColumnParallelLinearWithLoRA,
                                        LinearScalingRotaryEmbeddingWithLoRA,
                                        LogitsProcessorWithLoRA,
                                        MergedColumnParallelLinearWithLoRA,
                                        MergedQKVParallelLinearWithLoRA,
                                        QKVParallelLinearWithLoRA,
                                        RowParallelLinearWithLoRA,
                                        VocabParallelEmbeddingWithLoRA)

# yapf: enable  # noqa: ERA001

_all_lora_classes: set[type[BaseLayerWithLoRA]] = {
    VocabParallelEmbeddingWithLoRA,
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    QKVParallelLinearWithLoRA,
    MergedQKVParallelLinearWithLoRA,
    RowParallelLinearWithLoRA,
    LogitsProcessorWithLoRA,
    ColumnParallelLinearWithShardedLoRA,
    QKVParallelLinearWithShardedLoRA,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithShardedLoRA,
    RowParallelLinearWithShardedLoRA,
    LinearScalingRotaryEmbeddingWithLoRA,
}
