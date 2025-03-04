#!/usr/bin/env python3
# encoding: utf-8
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

from typing import List, Tuple

from vllm.sequence import IntermediateTensors

from vllm_mindspore.utils import get_valid_dtype

import mindspore as ms
from mindspore import mint


class PPMissingLayer(ms.nn.Cell):
    """
    A placeholder layer for missing layers in a pipeline parallel model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def construct(self, inputs):
        return inputs


def maybe_offload_to_cpu(module):
    # TODO: support
    return module


def maybe_prefix(prefix: str, name: str) -> str:
    """Add a prefix to a name if the prefix is non-empty.

    Args:
        prefix: The prefix to add. If empty, no prefix will be added.
        name: The name to potentially prefix.

    Returns:
        The string "prefix.name" if prefix was non-empty, otherwise just "name".
    """
    return name if not prefix else f"{prefix}.{name}"


def extract_layer_index(layer_name: str) -> int:
    """
    Extract the layer index from the module name.
    Examples:
    - "encoder.layers.0" -> 0
    - "encoder.layers.1.self_attn" -> 1
    - "2.self_attn" -> 2
    - "model.encoder.layers.0.sub.1" -> ValueError
    """
    subnames = layer_name.split(".")
    int_vals: List[int] = []
    for subname in subnames:
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
    assert len(int_vals) == 1, (
        f"layer name {layer_name} should" " only contain one integer"
    )
    return int_vals[0]


def make_layers(
    num_hidden_layers: int,
    layer_fn,
    prefix: str,
) -> Tuple[int, int, ms.nn.CellList]:
    """Make a list of layers with the given layer function, taking
    pipeline parallelism into account.
    """
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.distributed.utils import get_pp_indices

    start_layer, end_layer = get_pp_indices(
        num_hidden_layers, get_pp_group().rank_in_group, get_pp_group().world_size
    )
    modules = ms.nn.CellList(
        [PPMissingLayer() for _ in range(start_layer)]
        + [
            maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}"))
            for idx in range(start_layer, end_layer)
        ]
        + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
    )
    return start_layer, end_layer, modules


def make_empty_intermediate_tensors_factory(keys: List[str], hidden_size: int):

    def make_empty_intermediate_tensors(
        batch_size: int,
        dtype,
        device,
    ) -> IntermediateTensors:
        dtype = get_valid_dtype(dtype)
        return IntermediateTensors(
            {key: mint.zeros((batch_size, hidden_size), dtype=dtype) for key in keys}
        )

    return make_empty_intermediate_tensors
