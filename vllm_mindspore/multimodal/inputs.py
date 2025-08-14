# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/multimodal/inputs.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2025 The vLLM team.
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
"""Adaption for mindspore."""
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union, cast

import mindspore
from vllm.multimodal import MultiModalKwargs
from vllm.multimodal.inputs import (BaseMultiModalField, BatchedTensorInputs,
                                    JSONTree, json_map_leaves,
                                    nested_tensors_equal)

NestedTensors = Union[
    list["NestedTensors"],
    list[mindspore.Tensor],
    mindspore.Tensor,
    tuple[mindspore.Tensor, ...],
]


@dataclass
class MultiModalFieldElem:
    """
    Represents a keyword argument corresponding to a multi-modal item
    in :class:`MultiModalKwargs`.
    """

    modality: str
    """
    The modality of the corresponding multi-modal item.
    Each multi-modal item can consist of multiple keyword arguments.
    """

    key: str
    """
    The key of this field in :class:`MultiModalKwargs`,
    i.e. the name of the keyword argument to be passed to the model.
    """

    data: NestedTensors
    """
    The tensor data of this field in :class:`MultiModalKwargs`,
    i.e. the value of the keyword argument to be passed to the model.
    """

    field: "BaseMultiModalField"
    """
    Defines how to combine the tensor data of this field with others
    in order to batch multi-modal items together for model inference.
    """

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        return ((self.modality, self.key) == (other.modality, other.key)
                and nested_tensors_equal(self.data, other.data)
                and type(self.field) == type(other.field))  # noqa: E721


@staticmethod  # type: ignore
def as_kwargs(
    batched_inputs: BatchedTensorInputs,
    *,
    device=None,
) -> BatchedTensorInputs:
    # replace as_kwargs of vLLM for multi-model
    json_inputs = cast(JSONTree[mindspore.Tensor], batched_inputs)

    json_mapped = json_map_leaves(
        lambda x: x,
        json_inputs,
    )

    return cast(BatchedTensorInputs, json_mapped)


def from_items(items):
    """Construct a new :class:`MultiModalKwargs` from multiple items."""
    elems_by_key = defaultdict[str, list[MultiModalFieldElem]](list)
    for item in items:
        for key, elem in item.items():
            # transform elem.data to tensor, gpu is tensor.
            elem.data = mindspore.Tensor(elem.data)
            elems_by_key[key].append(elem)
    data = {
        key: elems[0].field.reduce_data(elems)
        for key, elems in elems_by_key.items() if len(elems) > 0
    }

    return MultiModalKwargs(data, items=items)


def build_elems(
    self,
    modality: str,
    key: str,
    data: NestedTensors,
) -> Sequence[MultiModalFieldElem]:
    field_factory = self._field_factory(modality=modality, key=key)
    return [field_factory(data[cast(slice, s)]) for s in self.slices]
