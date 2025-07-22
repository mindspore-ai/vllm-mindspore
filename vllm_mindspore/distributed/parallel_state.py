# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional, Union

import torch
import torch.distributed
from torch.distributed import ProcessGroup
from vllm.distributed.parallel_state import TensorMetadata, _split_tensor_dict

from vllm_mindspore.utils import atlas_inference


def gc_broadcast_tensor_dict(
    self,
    tensor_dict: Optional[dict[str, Union[torch.Tensor, Any]]] = None,
    src: int = 0,
    group: Optional[ProcessGroup] = None,
    metadata_group: Optional[ProcessGroup] = None
) -> Optional[dict[str, Union[torch.Tensor, Any]]]:
    """Broadcast the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.
        """
    # Bypass the function if we are using only 1 GPU.
    if (not torch.distributed.is_initialized() or self.world_size == 1):
        return tensor_dict

    if not atlas_inference():
        group = self.device_group
    metadata_group = self.cpu_group
    assert src < self.world_size, f"Invalid src rank ({src})"

    rank_in_group = self.rank_in_group
    if rank_in_group == src:
        metadata_list: list[tuple[Any, Any]] = []
        assert isinstance(
            tensor_dict,
            dict), (f"Expecting a dictionary, got {type(tensor_dict)}")
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        # `metadata_list` lives in CPU memory.
        # `broadcast_object_list` has serialization & deserialization,
        # all happening on CPU. Therefore, we can use the CPU group.
        self.broadcast_object(metadata_list, src=src)
        async_handles = []
        for tensor in tensor_list:
            if tensor.numel() == 0:
                # Skip broadcasting empty tensors.
                continue
            if tensor.is_cpu:
                # use metadata_group for CPU tensors
                handle = torch.distributed.broadcast(tensor,
                                                     src=self.ranks[src],
                                                     group=metadata_group,
                                                     async_op=True)
            else:
                # use group for GPU tensors
                handle = torch.distributed.broadcast(tensor,
                                                     src=self.ranks[src],
                                                     group=group,
                                                     async_op=True)
            async_handles.append(handle)
        for async_handle in async_handles:
            async_handle.wait()

    else:
        metadata_list = self.broadcast_object(None, src=src)
        tensor_dict = {}
        async_handles = []
        for key, value in metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(value.size,
                                     dtype=value.dtype,
                                     device=value.device)
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    tensor_dict[key] = tensor
                    continue
                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    handle = torch.distributed.broadcast(tensor,
                                                         src=self.ranks[src],
                                                         group=metadata_group,
                                                         async_op=True)
                else:
                    # use group for GPU tensors
                    handle = torch.distributed.broadcast(tensor,
                                                         src=self.ranks[src],
                                                         group=group,
                                                         async_op=True)
                async_handles.append(handle)
                tensor_dict[key] = tensor
            else:
                tensor_dict[key] = value
        for async_handle in async_handles:
            async_handle.wait()
    return tensor_dict
