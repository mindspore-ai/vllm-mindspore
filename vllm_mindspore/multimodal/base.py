# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/multimodal/base.py
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
"""adapt for optimization of multimodal data init"""
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vllm.sequence import SequenceGroupMetadata

from vllm.multimodal.base import MultiModalPlaceholderMap
from vllm.multimodal.inputs import MultiModalKwargs


@classmethod
def from_seq_group(
    cls, seq_group: "SequenceGroupMetadata", positions: range
) -> tuple[Optional[MultiModalKwargs], dict[str, "MultiModalPlaceholderMap"]]:
    """
    Returns the multi-modal items that intersect with the portion of a
    prompt (``seq_group``) represented by ``positions``, as well as a
    ``MultiModalPlaceholderMap`` that relates the multi-modal embedding
    vectors to their corresponding placeholders.

    Examples:

    ```
    Prompt:    |AAAA BBBB What's in these images?|
    Positions: |.................................|

        images      = [A, B]
        src_ranges  = [(0, 4), (4, 8)]
        dest_ranges = [(0, 4), (5, 9)]

    Prompt:    |AAAA BBBB What's in these images?|
    Positions: |  .....                          |

        images      = [A, B]
        src_ranges  = [(2, 4), (4, 6)]
        dest_ranges = [(0, 2), (3, 5)]

    Prompt:    |AAAA BBBB What's in these images?|
    Positions: |     .........                   |

        images      = [B]
        src_ranges  = [(0, 4)]
        dest_ranges = [(0, 4)]

    Prompt:    |AAAA BBBB What's in these images?|
    Positions: |          .......................|

        images      = []
        src_ranges  = []
        dest_ranges = []
    ```
    """
    seq_mm_data = seq_group.multi_modal_data
    seq_mm_placeholders = seq_group.multi_modal_placeholders

    if not seq_mm_data or not seq_mm_placeholders:
        # Avoiding initializing useless multimodal data,
        # 'MultiModalKwargs({})' is replaced with 'seq_mm_data'
        # updating by vllm-mindspore plugin
        return seq_mm_data, {}

    placeholder_maps = dict[str, MultiModalPlaceholderMap]()

    for modality, placeholders in seq_mm_placeholders.items():
        placeholder_map = MultiModalPlaceholderMap()

        if positions:
            placeholder_map.append_items_from_seq_group(
                positions,
                # Dummy, since we don't care about intersecting items
                [None] * len(placeholders),
                placeholders,
            )

        placeholder_maps[modality] = placeholder_map

    return seq_mm_data, placeholder_maps
