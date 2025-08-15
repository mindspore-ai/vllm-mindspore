# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/model_executor/layers/layernorm.py
#
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
"""Custom normalization layers."""

from typing import Optional, Union

from mindspore import Parameter, Tensor, mint, nn, ops
from mindspore._c_expression import typing
from mindspore.ops.auto_generate.gen_ops_prim import AddRmsNorm
from vllm.config import get_current_vllm_config
from vllm_mindspore.model_executor.utils import get_model_context


class RMSNorm(nn.Cell):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        params_dtype: Optional[typing.Type] = None,
    ) -> None:
        super().__init__()
        if params_dtype is None:
            params_dtype = get_model_context("model_dtype")
        self.weight = Parameter(mint.ones(hidden_size, dtype=params_dtype))
        self.rms_norm = ops.RmsNorm(eps)
        self.eps = eps
        self.add_rms_norm = AddRmsNorm()

    def construct(
        self,
        x: Tensor,
        residual: Optional[Tensor] = None
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        if residual is not None:
            output, _, residual = self.add_rms_norm(x, residual, self.weight,
                                                    self.eps)
            return output, residual
        output = self.rms_norm(x, self.weight)[0]
        return output
