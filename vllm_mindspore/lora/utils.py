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


def replace_submodule(model, module_name, new_module):
    """Replace a submodule in a model with a new module.

    Sets name attributes for base_layer weights and biases.
    For LoRA weights, only sets names in graph mode (where they are
    Parameters). In eager mode, LoRA weights are tuples of Tensors and
    don't need name attributes.
    """
    from mindspore import Parameter

    parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
    target_name = module_name.split(".")[-1]
    setattr(parent, target_name, new_module)

    def _set_param_name(obj, attr, suffix):
        """Helper to set parameter name if attribute exists and is not None."""
        param = getattr(obj, attr, None)
        if param is not None:
            param.name = f"{module_name}.{suffix}"

    # Set base_layer weight and bias names
    _set_param_name(new_module.base_layer, 'weight', 'weight')
    _set_param_name(new_module.base_layer, 'bias', 'bias')

    # Set LoRA weight names (only in graph mode where they are Parameters)
    lora_params = [
        ('lora_a_stacked', 'lora_a_weight'),
        ('lora_b_stacked', 'lora_b_weight'),
        ('lora_bias_stacked', 'lora_bias'),
    ]
    for attr, suffix in lora_params:
        param = getattr(new_module, attr, None)
        if param is not None and isinstance(param, Parameter):
            param.name = f"{module_name}.{suffix}"

    return new_module
