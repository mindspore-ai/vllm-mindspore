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
"""Patching ray functions to use view(dtype)"""
import contextlib

import numpy as np
import torch
from mindspore import Tensor


@contextlib.contextmanager
def _view_adapter():
    """A context manager to patch Tensor.view(dtype)."""

    def _view(tensor, target_dtype):
        """ function for view(dtype) """
        ori_shape = tensor.shape
        target_shape = (-1, )
        if len(ori_shape) > 1:
            target_shape = ori_shape[:-1] + target_shape
        out = np.frombuffer(
            tensor.numpy(),
            torch.ops.creation._TypeDict.get(target_dtype, np.float32))
        if not out.flags.aligned:
            out = np.require(out, requirements=["ALIGNED"])
        if target_dtype == torch.bfloat16:
            return torch.tensor(out.astype(
                np.float32)).astype(target_dtype).reshape(target_shape)
        return torch.tensor(out).reshape(target_shape)

    ori_view = torch._tensor.view
    Tensor.view = _view
    try:
        yield
    finally:
        Tensor.view = ori_view


def wrapper(func):
    """A wrapper to use _view_adapter."""

    def new_func(*args, **kwargs):
        with _view_adapter():
            return func(*args, **kwargs)

    return new_func


def patch_ray():
    """patch for ray serialization context to use view(dtype) """
    try:
        from ray._version import version
        from ray.experimental.channel.serialization_context import (
            _SerializationContext)
        if version >= "2.47.0":
            _SerializationContext.deserialize_from_numpy_or_scalar = \
                wrapper(_SerializationContext.deserialize_from_numpy_or_scalar)
            _SerializationContext.serialize_to_numpy_or_scalar = \
                wrapper(_SerializationContext.serialize_to_numpy_or_scalar)
        else:
            _SerializationContext.deserialize_from_numpy = \
                wrapper(_SerializationContext.deserialize_from_numpy)
            _SerializationContext.serialize_to_numpy = \
                wrapper(_SerializationContext.serialize_to_numpy)
    except ImportError:
        pass
