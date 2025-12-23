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

from mindspore import Tensor, mint

_SAMPLING_EPS = 1e-5


def apply_temperature(
    self,
    logits: Tensor,
    temp: Tensor,
    all_random: bool,
) -> Tensor:
    # Use in-place division to avoid creating a new tensor.
    # Avoid division by zero if there are greedy requests.
    if not all_random:
        temp = mint.where(temp < _SAMPLING_EPS, 1.0, temp)

    # logits.div_ will cause some error right now.
    # So we use logits = logits.div instead of logits.div_.
    return logits.div(temp.unsqueeze(dim=1))
