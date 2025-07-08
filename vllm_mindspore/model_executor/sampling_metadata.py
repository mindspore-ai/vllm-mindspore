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

from array import array
from dataclasses import dataclass
from typing import List

from vllm.utils import (
    is_pin_memory_available,
    make_tensor_with_pad,
)

_SAMPLING_EPS = 1e-5

from mindspore import Tensor
import mindspore as ms


@dataclass
class SamplingTensors:
    """Tensors for sampling."""

    temperatures: Tensor
    top_ps: Tensor
    top_ks: Tensor
    min_ps: Tensor
    presence_penalties: Tensor
    frequency_penalties: Tensor
    repetition_penalties: Tensor
    prompt_tokens: Tensor
    output_tokens: Tensor

    @classmethod
    def from_lists(
        cls,
        temperatures: List[float],
        top_ps: List[float],
        top_ks: List[int],
        min_ps: List[float],
        presence_penalties: List[float],
        frequency_penalties: List[float],
        repetition_penalties: List[float],
        prompt_tokens: List[array],
        output_tokens: List[array],
        vocab_size: int,
        device,
        dtype,
    ) -> "SamplingTensors":
        # Note that the performance will be very bad without
        # pinned memory.
        pin_memory = is_pin_memory_available()

        do_penalties = prompt_tokens or output_tokens

        if do_penalties:
            prompt_t = make_tensor_with_pad(
                prompt_tokens,
                vocab_size,
                device="cpu",
                dtype=ms.int64,
                pin_memory=pin_memory,
            )
            output_t = make_tensor_with_pad(
                output_tokens,
                vocab_size,
                device="cpu",
                dtype=ms.int64,
                pin_memory=pin_memory,
            )
        else:
            # TODO: to support empty
            # empty_tensor = torch.empty(0, device=device, dtype=torch.long)
            empty_tensor = Tensor(0, dtype=ms.int64)
            prompt_t = empty_tensor
            output_t = empty_tensor

        temperatures_t = Tensor(
            temperatures,
            dtype=dtype,
        )
        top_ps_t = Tensor(
            top_ps,
            dtype=dtype,
        )
        min_ps_t = Tensor(
            min_ps,
            dtype=dtype,
        )
        presence_penalties_t = Tensor(
            presence_penalties,
            dtype=dtype,
        )
        frequency_penalties_t = Tensor(
            frequency_penalties,
            dtype=dtype,
        )
        repetition_penalties_t = Tensor(
            repetition_penalties,
            dtype=dtype,
        )
        top_ks_t = Tensor(
            top_ks,
            dtype=ms.int64,
        )
        # Because the memory is pinned, we can do non-blocking
        # transfer to device.

        # For MindSpore: MindSpore does not support to device now
        return cls(
            temperatures=temperatures_t,
            top_ps=top_ps_t,
            top_ks=top_ks_t,
            min_ps=min_ps_t,
            presence_penalties=presence_penalties_t,
            frequency_penalties=frequency_penalties_t,
            repetition_penalties=repetition_penalties_t,
            prompt_tokens=prompt_t,
            output_tokens=output_t,
        )
