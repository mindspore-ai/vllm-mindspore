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
from typing import List, Optional, Any

import numpy as np
import torch
import torch.distributed


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
