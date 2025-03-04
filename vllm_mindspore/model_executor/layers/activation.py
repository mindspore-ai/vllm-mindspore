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

from mindspore import Tensor, mint, nn, ops


class SiluAndMul(nn.Cell):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    def construct(self, x):
        d = x.shape[-1] // 2
        return mint.nn.functional.silu(x[..., :d]) * x[..., d:]


class SwiGLU(nn.Cell):
    """An activation function for SwiGLU.

    Shapes:
        x: (batch_size, seq_len, 2 * hidden_size)
        return: (batch_size, seq_len, hidden_size)
    """

    def __init__(self):
        super().__init__()
        self.silu = nn.SiLU()
        self.split = ops.auto_generate.SplitWithSize()
        self.mul = ops.Mul()

    def construct(self, x: Tensor) -> Tensor:
        hidden_size = x.shape[-1] // 2
        size = [hidden_size, hidden_size]
        gate, hidden = self.split(x, size, dim=-1)
        gate = self.silu(gate)
        hidden = self.mul(hidden, gate)
        return hidden
