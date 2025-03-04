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

from typing import Any, Dict, Optional
from mindspore import Tensor

# TODO(tronzhang): Use vllm's latter...


def set_weight_attrs(
    weight: Tensor,
    weight_attrs: Optional[Dict[str, Any]],
):
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        setattr(weight, key, value)
