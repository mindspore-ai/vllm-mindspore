# SPDX-License-Identifier: Apache-2.0

# Functions are adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/v1/sample/sampler.py
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

import torch


def apply_temperature(
    self,
    logits: torch.Tensor,
    temp: torch.Tensor,
    all_random: bool,
) -> torch.Tensor:
    # logits.div_ will cause some error right now.
    # So we use logits = logits.div instead of logits.div_.
    return logits.div(temp.unsqueeze(dim=1))
