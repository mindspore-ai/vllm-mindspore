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

import pickle
from typing import List, Optional, Any, Dict, Union

import numpy as np
import torch
import torch.distributed
from torch.distributed import Backend, ProcessGroup
from mindspore.common.api import _pynative_executor


def init_model_parallel_group(
    group_ranks: List[List[int]],
    local_rank: int,
    backend: str,
    use_custom_allreduce: Optional[bool] = None,
    use_message_queue_broadcaster: bool = False,
    group_name: Optional[str] = None,
) -> "GroupCoordinator":
    from vllm.distributed.parallel_state import (
        GroupCoordinator,
        _ENABLE_CUSTOM_ALL_REDUCE,
    )

    if use_custom_allreduce is None:
        use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE

    # TODO(tronzhang): mindspore doesnot support enough communicate cpu ops, set use_message_queue_broadcaster to False now.
    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=False,
        use_custom_allreduce=use_custom_allreduce,
        use_tpu_communicator=True,
        use_hpu_communicator=True,
        use_xpu_communicator=True,
        use_message_queue_broadcaster=False,
        group_name=group_name,
    )


def all_reduce_for_GroupCoordinator(self, input_: torch.Tensor) -> torch.Tensor:
    """
    User-facing all-reduce function before we actually call the
    all-reduce operation.

    We need this because Dynamo does not support passing an arbitrary
    object (`self` in this case) to a custom op. We need to pass the
        group name as a string, and then look up the group coordinator from
        the group name, dispatch the all-reduce operation to the group
        coordinator.

    In addition, PyTorch custom ops do not support mutation or returning
    a new tensor in the same op. So we always make the all-reduce operation
    out-of-place.
    """
    # Bypass the function if we are using only 1 GPU.
    if self.world_size == 1:
        return input_

    torch.distributed.all_reduce(input_, group=self.device_group)
    return input_

def broadcast_tensor_dict(
    self,
    tensor_dict: Optional[Dict[str, Union[torch.Tensor, Any]]] = None,
    src: int = 0,
    group: Optional[ProcessGroup] = None,
    metadata_group: Optional[ProcessGroup] = None
) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
    """Broadcast the input tensor dictionary.
    NOTE: `src` is the local rank of the source rank.
    """
    # sync and start spin thread
    _pynative_executor.sync()
    _pynative_executor.set_async_for_graph(True)
        
    # Bypass the function if we are using only 1 GPU.
    if (not torch.distributed.is_initialized() or self.world_size == 1):
        return tensor_dict

    group = self.device_group
    metadata_group = self.cpu_group
    assert src < self.world_size, f"Invalid src rank ({src})"

    rank_in_group = self.rank_in_group
    if rank_in_group == src:
        metadata_list: List[Tuple[Any, Any]] = []
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
                    handle = torch.distributed.broadcast(
                        tensor,
                        src=self.ranks[src],
                        group=metadata_group,
                        async_op=True)
                else:
                    # use group for GPU tensors
                    handle = torch.distributed.broadcast(
                        tensor,
                        src=self.ranks[src],
                        group=group,
                        async_op=True)
                async_handles.append(handle)
                tensor_dict[key] = tensor
            else:
                tensor_dict[key] = value
        for async_handle in async_handles:
            async_handle.wait()
    _pynative_executor.sync()
    _pynative_executor.set_async_for_graph(False)
    return tensor_dict

