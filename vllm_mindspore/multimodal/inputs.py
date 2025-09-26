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
import numpy as np
import torch
from vllm.multimodal import MultiModalKwargs
from vllm.multimodal.inputs import (BaseMultiModalField, BatchedTensorInputs,
                                    JSONTree, is_list_of, json_map_leaves,
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


def flat_build_elems(
    self,
    modality: str,
    key: str,
    data: NestedTensors,
) -> Sequence[MultiModalFieldElem]:
    field_factory = self._field_factory(modality=modality, key=key)
    return [field_factory(data[cast(slice, s)]) for s in self.slices]


def batched_reduce_data(self, batch: list[NestedTensors]) -> NestedTensors:
    # NOTE: vLLM-MindSpore Plugin:
    # Currently mindspore does not support operating tensors in a
    # multi-threaded environment, so convert tensors to numpy.
    if len(batch) > 0 and is_list_of(batch, torch.Tensor, check="all"):
        if len(batch) == 1:
            # An optimization when `batch` contains only one tensor:
            # - produce exactly same result as `torch.stack(batch)`
            # - will achieve zero-copy if the tensor is contiguous
            return mindspore.from_numpy(np.expand_dims(batch[0].numpy(), 0))
        first_shape = batch[0].shape
        if all(elem.shape == first_shape for elem in batch):
            return mindspore.from_numpy(np.stack([b.numpy() for b in batch]))

    return batch


def flat_reduce_data(self, batch: list[NestedTensors]) -> NestedTensors:
    # NOTE: vLLM-MindSpore Plugin:
    # Currently mindspore does not support operating tensors in a
    # multi-threaded environment, so convert tensors to numpy.
    if len(batch) > 0 and is_list_of(batch, torch.Tensor, check="all"):
        if len(batch) == 1:
            # An optimization when `batch` contains only one tensor:
            # - produce exactly same result as `torch.concat(batch)`
            # - will achieve zero-copy if the tensor is contiguous
            return mindspore.from_numpy(batch[0].numpy())

        def _expect_same_shape(tensor: torch.Tensor):
            return tensor.shape[:self.dim] + tensor.shape[self.dim + 1:]

        first_shape = _expect_same_shape(batch[0])

        if all(_expect_same_shape(elem) == first_shape for elem in batch):
            return mindspore.from_numpy(
                np.concatenat([b.numpy() for b in batch], axis=self.dim))

    assert self.dim == 0, "dim == 0 is required for nested list"
    return [e for elem in batch for e in elem]


@staticmethod
def _try_stack(nested_tensors: NestedTensors,
               pin_memory: bool = False) -> NestedTensors:
    """
    Stack the inner dimensions that have the same shape in
    a nested list of tensors.

    Thus, a dimension represented by a list means that the inner
    dimensions are different for each element along that dimension.
    """
    if isinstance(nested_tensors, torch.Tensor):
        return nested_tensors

    # TODO: Remove these once all models have been migrated
    if isinstance(nested_tensors, np.ndarray):
        return torch.from_numpy(nested_tensors)
    if isinstance(nested_tensors, (int, float)):
        return torch.tensor(nested_tensors)

    stacked = [
        MultiModalKwargs._try_stack(t, pin_memory) for t in nested_tensors
    ]
    if not is_list_of(stacked, torch.Tensor, check="all"):
        # Only tensors (not lists) can be stacked.
        return stacked

    # NOTE: vLLM-MindSpore Plugin:
    # Currently mindspore does not support operating tensors in a
    # multi-threaded environment, so convert tensors to numpy.
    tensors_ = cast(list[torch.Tensor], stacked)
    if len(tensors_) == 1:
        # An optimization when `tensors_` contains only one tensor:
        # - produce exactly same result as `torch.stack(tensors_)`
        # - will achieve zero-copy if the tensor is contiguous
        return mindspore.from_numpy(np.expand_dims(tensors_[0].numpy(), 0))

    if any(t.shape != tensors_[0].shape for t in tensors_):
        # The tensors have incompatible shapes and can't be stacked.
        return tensors_

    return mindspore.from_numpy(np.stack([t.numpy() for t in tensors_]))
