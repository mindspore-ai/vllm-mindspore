# SPDX-License-Identifier: Apache-2.0

# Communication functions are adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/distributed/communication_op.py
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
"""
Implement a unified communication interface for both graph and pynative mode.
"""

from mindspore import nn, ops
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size, get_tp_group)


class ReduceFromModelParallelRegion(nn.Cell):
    "All reduce the input from the model parallel region."

    def __init__(self):
        super().__init__()
        self.world_size = get_tensor_model_parallel_world_size()
        if self.world_size > 1:
            self.tp_group = get_tp_group().device_group._name
            self.all_reduce = ops.AllReduce(group=self.tp_group)

    def construct(self, input_):
        if self.world_size == 1:
            return input_
        output = self.all_reduce(input_)
        return output


class AllGatherFromModelParallelRegion(nn.Cell):
    """
    Gather the input from world parallel region and concatenate,
    simultaneously perform transpose operation on input.
    """

    def __init__(self):
        super().__init__()
        self.world_size = get_tensor_model_parallel_world_size()
        if self.world_size > 1:
            self.tp_group = get_tp_group().device_group._name
            self.all_gather_into_tensor = ops.AllGather(group=self.tp_group)

    def construct(self, input_):
        # Size and dimension.
        if self.world_size == 1:
            return input_
        input_ = ops.swapaxes(input_, 0, -1)
        output = self.all_gather_into_tensor(input_)
        output = ops.swapaxes(output, 0, -1)
        return output
