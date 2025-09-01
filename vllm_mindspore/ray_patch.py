# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/ray-project/ray/blob/ray-2.49.0/python/ray/experimental/channel/serialization_context.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2025 The Ray Authors.
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
import numpy as np
import torch

try:
    from ray.experimental.util.types import Device
except ImportError:
    Device = None


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


def serialize_to_numpy_or_scalar(self, tensor):
    """
    patch for
    ray.experimental.channel.serialization_context._SerializationContext
    """
    tensor_device_type = tensor.device.type
    if tensor_device_type != "cpu":
        tensor = tensor.to("cpu")
    if tensor.dim() > 0:
        return (_view(tensor,
                      torch.uint8).numpy(), tensor.dtype, tensor_device_type)
    else:
        return (tensor.item(), tensor.dtype, tensor_device_type)


def deserialize_from_numpy_or_scalar(self, np_array, dtype, tensor_device_type,
                                     target_device):
    """ 
    patch for 
    ray.experimental.channel.serialization_context._SerializationContext
    """

    if target_device == Device.DEFAULT:
        target_device_type = tensor_device_type
    elif target_device in [Device.GPU, Device.CUDA]:
        target_device_type = "cuda"
    else:
        target_device_type = target_device.value

    if target_device_type != "cpu":

        def convert_numpy_to_tensor(np_array):
            if not isinstance(np_array, np.ndarray):
                # For scalar tensors, create the 0-dim tensor.
                return torch.tensor(np_array,
                                    device=target_device_type,
                                    dtype=dtype)
            else:
                # For non-scalar tensors, view as the original dtype.
                # It does zero-copy convert np_array inside shared memory to
                # a tensor. Since we move data to GPU immediately, it is safe.
                cpu_tensor = torch.from_numpy(np_array)
                cpu_tensor = _view(cpu_tensor, dtype)
                return cpu_tensor.to(device=target_device_type)

        gpu_tensor = convert_numpy_to_tensor(np_array)

        return gpu_tensor

    if not isinstance(np_array, np.ndarray):
        # For scalar tensors, create the 0-dim tensor.
        return torch.tensor(np_array, device=target_device_type, dtype=dtype)
    else:
        # For non-scalar tensors, view as the original dtype.
        return _view(torch.tensor(np_array, device=target_device_type), dtype)


def patch_ray():
    """patch for ray serialization context to use view(dtype) """
    try:
        from ray._version import version
        from ray.experimental.channel.serialization_context import (
            _SerializationContext)
        if version >= "2.47.0":
            _SerializationContext.deserialize_from_numpy_or_scalar = \
                deserialize_from_numpy_or_scalar
            _SerializationContext.serialize_to_numpy_or_scalar = \
                serialize_to_numpy_or_scalar
        else:
            _SerializationContext.deserialize_from_numpy = \
                deserialize_from_numpy_or_scalar
            _SerializationContext.serialize_to_numpy = \
                serialize_to_numpy_or_scalar
    except ImportError:
        pass
