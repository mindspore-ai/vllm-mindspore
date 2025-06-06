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

from typing import Optional, Tuple, Union, Any

from mindspore import Parameter, Tensor, mint, ops
from mindspore.common import dtype as mstype
from mindspore.common.dtype import typing
from mindspore import nn

from vllm.config import get_current_vllm_config


class RMSNorm(nn.Cell):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        params_dtype: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if params_dtype is None:
            params_dtype = get_current_vllm_config().model_config.dtype
        self.weight = Parameter(mint.ones(hidden_size, dtype=params_dtype))
        self.rms_norm = ops.RmsNorm(eps)

    def construct(
        self,
        x: Tensor,
        residual: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if residual is not None:
            x = x + residual
            residual = x
        output = self.rms_norm(x, self.weight)[0]
        if residual is None:
            return output
        return output, residual
