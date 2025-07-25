# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Union

import mindspore as ms
import numpy as np
import torch
from msgspec import msgpack

mstype_str_to_np_type = {
    "Bool": np.bool_,
    "Int8": np.int8,
    "Uint8": np.uint8,
    "Int16": np.int16,
    "Uint16": np.uint16,
    "Int32": np.int32,
    "Uint32": np.uint32,
    "Int64": np.int64,
    "Uint64": np.uint64,
    "Float16": np.float16,
    "Float32": np.float32,
}


def _decode_tensor(self, arr: Any) -> torch.Tensor:
    dtype, shape, data = arr
    # Copy from inline representation, to decouple the memory storage
    # of the message from the original buffer. And also make Torch
    # not complain about a readonly memoryview.
    buffer = self.aux_buffers[data] if isinstance(data, int) \
        else bytearray(data)
    if not buffer:  # torch.frombuffer doesn't like empty buffers
        assert 0 in shape
        return torch.empty(shape, dtype=dtype)
    # Create uint8 array
    arr = torch.frombuffer(buffer, dtype=torch.uint8)
    # Convert back to proper shape & type
    arr = arr.numpy().view(dtype=mstype_str_to_np_type[dtype]).reshape(shape)
    tensor = ms.from_numpy(arr)
    return tensor


def _encode_tensor(
        self, obj: torch.Tensor
) -> tuple[str, tuple[int, ...], Union[int, memoryview]]:
    assert self.aux_buffers is not None
    # view the tensor as a contiguous 1D array of bytes
    # arr = obj.flatten().contiguous().view(torch.uint8).numpy()
    arr = obj.flatten().contiguous().numpy()
    arr = arr.view(dtype=np.uint8)
    if obj.nbytes < self.size_threshold:
        # Smaller tensors are encoded inline, just like ndarrays.
        CUSTOM_TYPE_RAW_VIEW = 3
        data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, arr.data)
    else:
        # Otherwise encode index of backing buffer to avoid copy.
        data = len(self.aux_buffers)
        self.aux_buffers.append(arr.data)
    dtype = str(obj.dtype).removeprefix("torch.")
    return dtype, obj.shape, data
