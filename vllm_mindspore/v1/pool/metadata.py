# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/pool/metadata.py
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
from vllm.v1.pool.metadata import PoolingCursor


def build_pooling_cursor(num_scheduled_tokens: list[int],
                         prompt_lens: torch.Tensor, device: torch.device):
    assert len(prompt_lens) == len(num_scheduled_tokens)

    n_seq = len(num_scheduled_tokens)
    index = list(range(n_seq))
    num_scheduled_tokens = torch.tensor(num_scheduled_tokens, device=device)
    cumsum = torch.zeros(n_seq, dtype=torch.int64, device=device)
    torch.cumsum(num_scheduled_tokens, dim=0, out=cumsum)
    # Out is not support for slice.
    cumsum = torch.cat((torch.tensor([0], dtype=torch.int64,
                                     device=device), cumsum))
    return PoolingCursor(index=index,
                         first_token_indices_gpu=cumsum[:n_seq],
                         last_token_indices_gpu=cumsum[1:] - 1,
                         prompt_lens_cpu=prompt_lens,
                         num_scheduled_tokens_cpu=num_scheduled_tokens)
