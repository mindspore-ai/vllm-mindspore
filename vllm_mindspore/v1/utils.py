# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/v1/utils.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2025 The vLLM team.
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

from typing import TYPE_CHECKING

import mindspore as ms
import numpy as np
import torch
from mindspore import Tensor
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_mindspore.model_executor.models.utils import convert_pin

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)


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


def create_dp_placement_groups(
        vllm_config: VllmConfig) -> tuple[list["PlacementGroup"], list[int]]:
    import ray
    from ray._private.state import available_resources_per_node
    from ray.util.state import list_nodes

    logger.info("Creating placement groups for data parallel")
    dp_master_ip = \
        vllm_config.parallel_config.data_parallel_master_ip
    dp_size = vllm_config.parallel_config.data_parallel_size
    local_engine_count = \
        vllm_config.parallel_config.data_parallel_size_local

    nodes = list_nodes()
    nodes = sorted(list_nodes(), key=lambda node: node.node_ip != dp_master_ip)
    assert nodes[0].node_ip == dp_master_ip, (
        "The first node must be the head node")
    assert len(nodes) == 1 or nodes[1].node_ip != dp_master_ip, (
        "There can only be one head node")

    available_resources = available_resources_per_node()
    world_size = vllm_config.parallel_config.world_size
    placement_groups: list[PlacementGroup] = []
    local_dp_ranks: list[int] = []

    for node in nodes:
        node_ip = node.node_ip
        node_resources = available_resources[node.node_id]
        # For now, each DP rank can only be assigned to one node
        # TODO(rui): support allocating a single DP rank
        # to multiple nodes

        # vllm-mindspore: Replace "GPU" with current_platform.ray_device_key to
        # support oot platform.
        from vllm.platforms import current_platform
        ray_device_key = current_platform.ray_device_key
        available_engine_count = int(
            node_resources[ray_device_key]) // world_size
        if node_ip == dp_master_ip:
            assert available_engine_count >= local_engine_count, (
                "Not enough resources to allocate DP ranks "
                f"on DP master node {node_ip}")
            for i in range(local_engine_count):
                bundles = [{
                    ray_device_key: 1.0,
                    "node:" + dp_master_ip: 0.001
                }] * world_size + [{
                    "CPU": 1.0
                }]
                pg = ray.util.placement_group(
                    name=f"dp_rank_{len(placement_groups)}",
                    strategy="STRICT_PACK",
                    bundles=bundles,
                )
                placement_groups.append(pg)
                local_dp_ranks.append(i)
        else:
            for i in range(available_engine_count):
                if len(placement_groups) == dp_size:
                    break
                bundles = [{ray_device_key: 1.0}] * world_size + [{"CPU": 1.0}]
                pg = ray.util.placement_group(
                    name=f"dp_rank_{len(placement_groups)}",
                    strategy="STRICT_PACK",
                    bundles=bundles,
                )
                placement_groups.append(pg)
                local_dp_ranks.append(i)
    return placement_groups, local_dp_ranks


class CpuGpuBuffer:
    """Buffer to easily copy tensors between CPU and GPU."""

    def __init__(
        self,
        *size,
        dtype,
        device,
        pin_memory: bool,
        with_numpy: bool = True,
    ) -> None:
        self.cpu = torch.zeros(*size,
                               dtype=dtype,
                               device="cpu",
                               pin_memory=pin_memory)
        # vllm-mindspore begin
        self.cpu = convert_pin(self.cpu)
        # vllm-mindspore end
        self.gpu = self.cpu.to(device)
        self.np: np.ndarray
        # To keep type hints simple (avoiding generics and subclasses), we
        # only conditionally create the numpy array attribute. This can cause
        # AttributeError if `self.np` is accessed when `with_numpy=False`.
        if with_numpy:
            if dtype == torch.bfloat16:
                raise ValueError(
                    "Bfloat16 torch tensors cannot be directly cast to a "
                    "numpy array, so call CpuGpuBuffer with with_numpy=False")
            self.np = self.cpu.numpy()

    def copy_to_gpu(self, n=None) -> torch.Tensor:
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=True)
        # vllm-mindspore begin
        return self.gpu[:n].copy_(ms.from_numpy(self.np[:n]),
                                  non_blocking=True)
        # vllm-mindspore end
