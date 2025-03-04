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
"""Attention backend utils"""

from contextlib import contextmanager
from typing import TYPE_CHECKING

from vllm.attention import AttentionState

if TYPE_CHECKING:
    from vllm.worker.model_runner_base import ModelRunnerBase


class MsAttentionState(AttentionState):

    def __init__(self, runner: "ModelRunnerBase"):
        self.runner = runner
        self._is_graph_capturing = False

    def begin_forward(self, model_input) -> None:
        return

    def get_graph_input_buffers(
        self, attn_metadata, is_encoder_decoder_model: bool = False
    ):
        """Get attention-specific input buffers for CUDA graph capture."""
        ...

    @contextmanager
    def graph_capture(self, max_batch_size: int):
        """Context manager used when capturing CUDA graphs."""
        yield

    def graph_capture_get_metadata_for_batch(
        self, batch_size: int, is_encoder_decoder_model: bool = False
    ): ...

    def graph_clone(self, batch_size: int): ...

    def prepare_graph_input_buffers(
        self, input_buffers, attn_metadata, is_encoder_decoder_model: bool = False
    ) -> None:
        """In-place modify input buffers dict for CUDA graph replay."""
        ...
