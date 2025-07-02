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

# the data type of finfo.tiny is not float but narray in msadapter,
# which is not supported to be a tensor index

from functools import cached_property
from typing import Dict

import mindtorch
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
    return float(mindtorch.torch.finfo(self.probs_dtype).tiny)
