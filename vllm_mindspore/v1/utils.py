# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologies Co., Ltd.
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

import mindspore as ms
import numpy as np
from mindspore import Tensor


def _copy_slice_from_np(from_np: np.ndarray, to_tensor: Tensor,
                        length: int) -> None:
    """
    Copy the first length elements of a numpy array into a tensor in a
    non-blocking manner.
    """
    to_tensor[:length] = ms.from_numpy(from_np[:length])
    return to_tensor


def copy_slice(from_tensor: Tensor,
               to_tensor: Tensor,
               length: int,
               *,
               return_tensor=True) -> None:
    """
    Copy the first length elements of a tensor into another tensor in a
    non-blocking manner.

    Used to copy pinned CPU tensor data to pre-allocated GPU tensors.
    """
    to_tensor[:length] = from_tensor[:length]
    if return_tensor:
        return to_tensor[:length]

