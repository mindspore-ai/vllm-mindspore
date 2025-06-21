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

from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional, Tuple, Union

import mindspore as ms
from mindspore import mint, ops
from vllm.sequence import IntermediateTensors

from vllm_mindspore.multimodal.inputs import NestedTensors
from vllm_mindspore.utils import get_valid_dtype

WeightsMapping = Mapping[str, Optional[str]]
"""If a key maps to a value of `None`, the corresponding weight is ignored."""


@dataclass
class WeightsMapper:
    """Maps the name of each weight if they match the following patterns."""

    orig_to_new_substr: WeightsMapping = field(default_factory=dict)
    orig_to_new_prefix: WeightsMapping = field(default_factory=dict)
    orig_to_new_suffix: WeightsMapping = field(default_factory=dict)

    def _map_name(self, key: str) -> Optional[str]:
        for substr, new_key in self.orig_to_new_substr.items():
            if substr in key:
                if new_key is None:
                    return None

                key = key.replace(substr, new_key, 1)

        for prefix, new_key in self.orig_to_new_prefix.items():
            if key.startswith(prefix):
                if new_key is None:
                    return None

                key = key.replace(prefix, new_key, 1)

        for suffix, new_key in self.orig_to_new_suffix.items():
            if key.endswith(suffix):
                if new_key is None:
                    return None

                key = new_key.join(key.rsplit(suffix, 1))

        return key

    def apply(
        self, weights: Iterable[Tuple[str, ms.Tensor]]
    ) -> Iterable[Tuple[str, ms.Tensor]]:
        return ((out_name, data) for name, data in weights
                if (out_name := self._map_name(name)) is not None)


enforce_eager = False


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
    assert len(int_vals) == 1, (f"layer name {layer_name} should"
                                " only contain one integer")
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

    start_layer, end_layer = get_pp_indices(num_hidden_layers,
                                            get_pp_group().rank_in_group,
                                            get_pp_group().world_size)
    modules = ms.nn.CellList([PPMissingLayer() for _ in range(start_layer)] + [
        maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}"))
        for idx in range(start_layer, end_layer)
    ] + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)])
    return start_layer, end_layer, modules


def make_empty_intermediate_tensors_factory(keys: List[str], hidden_size: int):

    def make_empty_intermediate_tensors(
        batch_size: int,
        dtype,
        device,
    ) -> IntermediateTensors:
        dtype = get_valid_dtype(dtype)
        return IntermediateTensors({
            key: mint.zeros((batch_size, hidden_size), dtype=dtype)
            for key in keys
        })

    return make_empty_intermediate_tensors


########################### for multi model ###########################


def _flatten_embeddings(embeddings: NestedTensors) -> ms.Tensor:
    """
    Recursively flattens and concatenates NestedTensors on all but the last
    dimension.
    """

    if isinstance(embeddings, ms.Tensor):
        # Flatten all but the last dimension.
        return embeddings.flatten(0, -2)

    return ops.cat(tuple(_flatten_embeddings(t) for t in embeddings))


def _embedding_count_expression(embeddings: NestedTensors) -> str:
    """
    Constructs a debugging representation of the number of embeddings in the
    NestedTensors.
    """

    if isinstance(embeddings, ms.Tensor):
        return " x ".join([str(dim) for dim in embeddings.shape[:-1]])

    return " + ".join(
        _embedding_count_expression(inner) for inner in embeddings)


def _merge_multimodal_embeddings(
    inputs_embeds: ms.Tensor,
    is_multimodal: ms.Tensor,
    multimodal_embeddings: NestedTensors,
) -> ms.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.

    Note:
        This updates ``inputs_embeds`` in place.
    """
    num_expected_tokens = is_multimodal.sum().item()
    assert isinstance(num_expected_tokens, int)

    flattened = _flatten_embeddings(multimodal_embeddings)
    if flattened.shape[0] != num_expected_tokens:
        expr = _embedding_count_expression(multimodal_embeddings)
        raise ValueError(
            f"Attempted to assign {expr} = {flattened.shape[0]} "
            f"multimodal tokens to {num_expected_tokens} placeholders")

    inputs_embeds[is_multimodal] = flattened
    return inputs_embeds


def merge_multimodal_embeddings(
    input_ids: ms.Tensor,
    inputs_embeds: ms.Tensor,
    multimodal_embeddings: NestedTensors,
    placeholder_token_id: Union[int, List[int]],
) -> ms.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.
    
    ``placeholder_token_id`` can be a list of token ids (e.g, token ids 
    of img_start, img_break, and img_end tokens) when needed: This means 
    the order of these tokens in the ``input_ids`` MUST MATCH the order of 
    their embeddings in ``multimodal_embeddings`` since we need to 
    slice-merge instead of individually scattering.

    For example, if input_ids is "TTTTTSIIIBIIIBIIIETTT", where
    - T is text token
    - S is image start token
    - I is image embedding token
    - B is image break token
    - E is image end token.
    
    Then the image embeddings (that correspond to I's) from vision encoder 
    must be padded with embeddings of S, B, and E in the same order of 
    input_ids for a correct embedding merge.

    Note:
        This updates ``inputs_embeds`` in place.
    """
    if isinstance(placeholder_token_id, list):
        placeholder_token_id = ms.Tensor(placeholder_token_id,
                                         device=input_ids.device)
        return _merge_multimodal_embeddings(
            inputs_embeds,
            ms.numpy.isin(input_ids, placeholder_token_id),
            multimodal_embeddings,
        )

    return _merge_multimodal_embeddings(
        inputs_embeds,
        (input_ids == placeholder_token_id),
        multimodal_embeddings,
    )
