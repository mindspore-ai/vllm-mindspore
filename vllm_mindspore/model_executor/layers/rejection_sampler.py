# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/model_executor/layers/rejection_sampler.py
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
"""Adaption for rejection sampler."""

# the data type of finfo.tiny is not float but narray in msadapter,
# which is not supported to be a tensor index

from functools import cached_property
from typing import Dict

import torch
import mindspore as ms

from vllm.platforms import current_platform

@cached_property
def _smallest_positive_value(self) -> float:
    """Return the smallest positive value representable by the probs dtype.
    This value is used when constructing a distribution from which to sample
    recovered tokens in the first rejection case.

    See _get_recovered_probs for more details

    Note that this isn't actually the smallest positive value representable
    by float32, but the smallest positive normal value.
    See https://en.wikipedia.org/wiki/Subnormal_number for more information.
    """
    # the value type of tiny is numpy in msadapter.
    return float(torch.finfo(self.probs_dtype).tiny)


# msadapter does not support 'exponential_'
@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def _multinomial(
        probs: torch.Tensor,
        num_samples: int,
        k: int,
        seeded_seqs: Dict[int, torch.Generator],
) -> torch.Tensor:
    # msadapter donot support tensor.exponential_
    def exponential_(x: torch.Tensor, lambda_, generator=None):
        random_x = ms.mint.rand(x.shape, generator=generator)  # 生成均匀分布随机数
        return -torch.log(random_x) / lambda_  # 逆变换法

    if num_samples > 1:
        # This is equivalent to torch.repeat_interleaved (which also
        # forces a GPU<->CPU sync).
        probs = probs[:, None, :].expand(probs.shape[0], num_samples,
                                         probs.shape[1]).contiguous().view(
            -1, probs.shape[1])
    q = torch.empty_like(probs)
    if not seeded_seqs:
        q = exponential_(q, 1.0)
    else:
        start = 0
        for idx in range(len(q) // k):
            end = start + k
            generator = seeded_seqs.get(idx)
            # Note: generator might be None for non seeded
            q[start:end] = exponential_(q[start:end], 1.0, generator=generator)
            start = end

    return probs.div_(q).argmax(dim=1).view(-1, num_samples)
