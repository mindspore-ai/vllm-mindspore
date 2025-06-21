#!/usr/bin/env python3
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from typing import Set, Type

from vllm.lora.fully_sharded_layers import (
    ColumnParallelLinearWithShardedLoRA,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithShardedLoRA, QKVParallelLinearWithShardedLoRA,
    RowParallelLinearWithShardedLoRA)

from vllm_mindspore.lora.layers import (
    BaseLayerWithLoRA, ColumnParallelLinearWithLoRA,
    LinearScalingRotaryEmbeddingWithLoRA, LogitsProcessorWithLoRA,
    MergedColumnParallelLinearWithLoRA, MergedQKVParallelLinearWithLoRA,
    QKVParallelLinearWithLoRA, RowParallelLinearWithLoRA,
    VocabParallelEmbeddingWithLoRA)

_all_lora_classes: Set[Type[BaseLayerWithLoRA]] = {
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
