# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/v1/worker/block_table.py
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

from typing import Union

import mindspore as ms
import numpy as np
import torch
from mindspore import Tensor
from vllm.distributed import get_dcp_group
from vllm.logger import init_logger
from vllm.v1.utils import CpuGpuBuffer

logger = init_logger(__name__)


class BlockTable:

    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device,
    ):
        self.block_size = block_size
        self.max_num_reqs = max_num_reqs
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device
        self.block_table = self._make_buffer(max_num_reqs,
                                             max_num_blocks_per_req,
                                             dtype=ms.int32)
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)
        self.slot_mapping = self._make_buffer(
            self.max_num_batched_tokens,
            # vllm-mindspore begin
            # for reshapeandcache's input, slot_mapping should be int32
            dtype=ms.int32)
        # vllm-mindspore end
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0

    def append_row(
        self,
        block_ids: list[int],
        row_idx: int,
    ) -> None:
        if not block_ids:
            return
        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        self.block_table.np[row_idx, start:start + num_blocks] = block_ids

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        block_table_np = self.block_table.np
        block_table_np[tgt, :num_blocks] = block_table_np[src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks

    def swap_row(self, src: int, tgt: int) -> None:
        src_tgt, tgt_src = [src, tgt], [tgt, src]
        self.num_blocks_per_row[src_tgt] = self.num_blocks_per_row[tgt_src]
        self.block_table.np[src_tgt] = self.block_table.np[tgt_src]

    def compute_slot_mapping(self, req_indices: np.ndarray,
                             positions: np.ndarray) -> None:
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size`
        # here because M (max_model_len) is not necessarily divisible by
        # block_size.
        if self.dcp_world_size > 1:
            # Note(hc): The DCP implement store kvcache with an interleave
            # style, the kvcache for the token whose token_idx is i is
            # always stored on the GPU whose dcp_rank equals i % cp_world_size:

            # Use a "virtual block" which equals to world_size * block_size
            # for block_table_indices calculation.
            virtual_block_size = self.block_size * self.dcp_world_size
            block_table_indices = (req_indices * self.max_num_blocks_per_req +
                                   positions // virtual_block_size)
            block_numbers = self.block_table.np.ravel()[block_table_indices]
            # Use virtual_block_size for mask calculation, which marks local
            # tokens.
            virtual_block_offsets = positions % virtual_block_size
            mask = virtual_block_offsets % self.dcp_world_size == self.dcp_rank
            # Calculate local block_offsets
            block_offsets = virtual_block_offsets // self.dcp_world_size
            # Calculate slot_mapping
            slot_mapping = block_numbers * self.block_size + block_offsets
            # Write final slots, use -1 for not-local
            self.slot_mapping.np[:req_indices.shape[0]] = np.where(
                mask, slot_mapping, -1)
        else:
            block_table_indices = (req_indices * self.max_num_blocks_per_req +
                                   positions // self.block_size)
            block_numbers = self.block_table.np.ravel()[block_table_indices]
            block_offsets = positions % self.block_size
            np.add(block_numbers * self.block_size,
                   block_offsets,
                   out=self.slot_mapping.np[:req_indices.shape[0]])

    def commit_block_table(self, num_reqs: int) -> None:
        self.block_table.copy_to_gpu(num_reqs)

    def commit_slot_mapping(self, num_tokens: int) -> None:
        self.slot_mapping.copy_to_gpu(num_tokens)

    def clear(self) -> None:
        self.block_table.gpu.fill_(0)
        self.block_table.cpu.fill_(0)

    def get_device_tensor(self, num_reqs: int) -> Tensor:
        """Returns the device tensor of the block table."""
        return self.block_table.gpu[:num_reqs]

    def get_cpu_tensor(self) -> Tensor:
        """Returns the CPU tensor of the block table."""
        return self.block_table.cpu

    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table."""
        return self.block_table.np

    def _make_buffer(self, *size: Union[int, torch.SymInt],
                     dtype: torch.dtype) -> CpuGpuBuffer:
        return CpuGpuBuffer(*size,
                            dtype=dtype,
                            device=self.device,
                            pin_memory=self.pin_memory)
