# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.11.0/tests/v1/sample/utils.py

# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2024-2025 The vLLM team.
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
'''
Logits processor utils
'''

from collections.abc import Iterator
from enum import Enum
from typing import NamedTuple, Optional

import vllm_mindspore
import torch
from vllm.utils import make_tensor_with_pad
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata


class BatchLogprobsComposition(Enum):
    """Types of logprobs configs to include in test batch"""
    NONE = 0
    SAMPLE = 1
    PROMPT = 2
    SAMPLE_PROMPT = 3


BatchLogprobsSpecType = list[tuple[Optional[int], Optional[int]]]


def create_fake_logits(batch_size: int, vocab_size: int) -> torch.Tensor:
    fake_logits = torch.full((batch_size, vocab_size), 1e-2, dtype=torch.float)
    return fake_logits


def create_penalty_tensor(batch_size: int, penalty_value: float,
                          device: torch.device) -> torch.Tensor:
    return torch.full((batch_size, ),
                      fill_value=penalty_value,
                      dtype=torch.float,
                      device=device)


def create_prompt_tokens_tensor(
    prompt_token_ids: list[list[int]],
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    return make_tensor_with_pad(
        prompt_token_ids,
        pad=vocab_size,
        device=device,
        dtype=torch.int64,
        pin_memory=False,
    )


class LogitsprocsTestFakes(NamedTuple):
    """Wraps fake data structures to support testing"""
    logits: torch.Tensor
    sampling_metadata: SamplingMetadata

    def get_logitsprocs_by_cls(
        self,
        cls: type[LogitsProcessor],
    ) -> Iterator[LogitsProcessor]:
        """Yield logits processors of a specific class.
        
        Args:
          cls: :class:`LogitsProcessor` subclass

        Returns:
          Iterator over logits processors
        """
        return (lp for lp in self.sampling_metadata.logitsprocs.all
                if isinstance(lp, cls))

    def get_logitsprocs(self) -> Iterator[LogitsProcessor]:
        """Iterator over all logits processors."""
        return self.sampling_metadata.logitsprocs.all


def fake_update_logitsprocs_state(
    test_fakes: LogitsprocsTestFakes,
    batch_update: BatchUpdate,
) -> None:
    """Imitate logits processors persistent batch state update
    in engine core"""
    for logitproc in test_fakes.get_logitsprocs():
        logitproc.update_state(batch_update)


def fake_apply_logitsprocs(
    test_fakes: LogitsprocsTestFakes,
    slice_indices: list[int],
) -> torch.Tensor:
    """Imitate application of logits processors in engine core"""
    logits = test_fakes.logits[torch.tensor(slice_indices,
                                            dtype=torch.long)].clone()
    for processor in test_fakes.get_logitsprocs():
        logits = processor.apply(logits)
    return logits
