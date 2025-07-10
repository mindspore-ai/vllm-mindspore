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
from typing import Union, cast

import mindspore

from vllm.multimodal.inputs import BatchedTensorInputs, JSONTree, json_map_leaves


NestedTensors = Union[list["NestedTensors"], list[mindspore.Tensor], mindspore.Tensor,
                      tuple[mindspore.Tensor, ...]]


@staticmethod
def as_kwargs(
    batched_inputs: BatchedTensorInputs,
    *,
    device = None,
) -> BatchedTensorInputs:
    # replace as_kwargs of vLLM for multi-model
    json_inputs = cast(JSONTree[mindspore.Tensor], batched_inputs)

    json_mapped = json_map_leaves(
        lambda x: x,
        json_inputs,
    )

    return cast(BatchedTensorInputs, json_mapped)