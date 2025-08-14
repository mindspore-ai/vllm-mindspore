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

import msadapter


# Override _smallest_positive_value in RejectionSampler to resolve
# the return type mismatch between our implementation and the PyTorch library.
@cached_property  # type: ignore[misc]
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
    return float(msadapter.finfo(self.probs_dtype).tiny)
