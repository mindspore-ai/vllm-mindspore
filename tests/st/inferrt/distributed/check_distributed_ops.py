# Copyright 2025 Huawei Technologies Co., Ltd
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
import os
from typing import List

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch._C._distributed_c10d import _resolve_process_group

import mrt
from mrt import jit

from mrt.collective import CollectiveManager
from mrt.torch import backend


import pytest
from tests.mark_utils import arg_mark

BACKEND_HCCL = "hccl"

def setup_distributed():
    rank = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    torch.npu.set_device(local_rank)

    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '6689')
    
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['RANK_ID'] = str(rank)
    
    dist.init_process_group(
        BACKEND_HCCL,
        rank=rank,
        world_size=world_size,
        init_method='env://'
    )


def cleanup():
    dist.destroy_process_group()


def check_group_info(pg=None):
    ptd = pg.group_name
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.getenv('LOCAL_RANK', "0"))
    world_size = dist.get_world_size()

    group_rank = dist.get_rank(pg)
    rank_list = dist.get_process_group_ranks(pg)
    group_size = dist.get_world_size(pg)

    group_rank_id = CollectiveManager.instance().get_group_rank(f"{ptd}")
    group_rank_size = CollectiveManager.instance().get_group_size(f"{ptd}")

    assert CollectiveManager.instance().global_rank_id() == rank, f"got global_rank_id {CollectiveManager.instance().global_rank_id()}, but expected {rank}"
    assert CollectiveManager.instance().local_rank_id() == int(local_rank), f"got local_rank_id {CollectiveManager.instance().local_rank_id()}, but expected {local_rank}"
    assert CollectiveManager.instance().global_rank_size() == world_size, f"got global_rank_size {CollectiveManager.instance().global_rank_size()}, but expected {world_size}"
    assert group_rank_id == group_rank
    assert group_rank_size == group_size
    return


def test_all_gather():
    """
    Feature: Check all_gather op launch
    Description: Check all_gather op launch with cache
    Expectation: The result is correct
    """
    class SimpleNetwork(nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x, gathered, pg=None):
            dist.all_gather_into_tensor(gathered, x, group=pg)
            output = torch.mul(gathered, gathered)
            return output

    def check_allgather_output(output, expect_output):
        output = output.cpu()
        expect_output = (expect_output * expect_output).cpu()
        assert (output == expect_output).all(), f"expected output is {expect_output}, but got {output}"

    setup_distributed()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size()
    group_list = [ii for ii in range(world_size)]
    new_pg = dist.new_group(group_list)

    model = SimpleNetwork().npu()
    example_input = torch.tensor([1,2]).npu() * (rank+1) * 10

    compiled_model = torch.compile(
        model,
        backend=backend,
        fullgraph=True,
    )

    world_size = dist.get_world_size(new_pg)
    gathered = [torch.zeros_like(example_input) for _ in range(world_size)]
    gathered = torch.cat(gathered, dim=0)
    expect_output = torch.zeros_like(gathered)
    dist.all_gather_into_tensor(expect_output, example_input, group=new_pg)

    output = compiled_model(example_input, gathered, new_pg)

    check_group_info(new_pg)
    check_allgather_output(output, expect_output)
    dist.destroy_process_group()


def test_reduce_scatter():
    """
    Feature: Check reduce_scatter op launch
    Description: Check reduce_scatter op launch with cache
    Expectation: The result is correct
    """
    class SimpleNetwork(nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, tensor_out, tensor_in, pg=None):
            dist.reduce_scatter_tensor(tensor_out, tensor_in, group=pg)
            output = torch.mul(tensor_out, tensor_out)
            return output

    def check_reduce_scatter_output(output, expect_out):
        output = output.cpu()
        expect_output = (expect_out * expect_out).cpu()
        assert (output == expect_output).all(), f"expected output is {expect_output}, but got {output}"

    setup_distributed()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size()
    group_list = [ii for ii in range(world_size)]
    new_pg = dist.new_group(group_list)

    model = SimpleNetwork().npu()
    # rank 0: [0, 1, 2, 3], rank 1: [0, 2, 4, 6]
    tensor_in = torch.arange(world_size * 2, dtype=torch.int64).npu() * (rank + 1)
    tensor_out = torch.zeros(2, dtype=torch.int64).npu()
    expect_out = torch.zeros(2, dtype=torch.int64).npu()
    dist.reduce_scatter_tensor(expect_out, tensor_in, group=new_pg)

    compiled_model = torch.compile(
        model,
        backend=backend,
        fullgraph=True,
    )

    world_size = dist.get_world_size(new_pg)
    output = compiled_model(tensor_out, tensor_in, new_pg)

    check_group_info(new_pg)
    check_reduce_scatter_output(output, expect_out)
    dist.destroy_process_group()


def test_all_reduce():
    """
    Feature: Check all_reduce op launch
    Description: Check all_reduce op launch with cache
    Expectation: The result is correct
    """
    class SimpleNetwork(nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, tensor_in, pg=None):
            dist.all_reduce(tensor_in, group=pg)
            output = torch.mul(tensor_in, tensor_in)
            return output

    def check_all_reduce_output(output, expect_out):
        output = output.cpu()
        expect_output = (expect_out * expect_out).cpu()
        assert (output == expect_output).all(), f"expected output is {expect_output}, but got {output}"
    setup_distributed()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size()
    group_list = [ii for ii in range(world_size)]
    new_pg = dist.new_group(group_list)

    model = SimpleNetwork().npu()
    # rank 0: [0, 1, 2, 3], rank 1: [0, 2, 4, 6]
    tensor_in = torch.arange(world_size * 2, dtype=torch.int64).npu() * (rank + 1)
    expect_out = torch.arange(world_size * 2, dtype=torch.int64).npu() * (rank + 1)
    dist.all_reduce(expect_out, group=new_pg)

    compiled_model = torch.compile(
        model,
        backend=backend,
        fullgraph=True,
    )

    world_size = dist.get_world_size(new_pg)
    output = compiled_model(tensor_in, new_pg)

    check_group_info(new_pg)
    check_all_reduce_output(output, expect_out)
    dist.destroy_process_group()


def test_all_to_all():
    """
    Feature: Check all_to_all op launch
    Description: Check all_to_all op launch with cache
    Expectation: The result is correct
    """
    class SimpleNetwork(nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, tensor_out, tensor_in, pg=None):
            dist.all_to_all_single(tensor_out, tensor_in, group=pg)
            output = torch.mul(tensor_out, tensor_out)
            return output

    def check_all_to_all_output(output, expect_out):
        output = output.cpu()
        expect_output = (expect_out * expect_out).cpu()
        assert (output == expect_output).all(), f"expected output is {expect_output}, but got {output}"

    setup_distributed()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size()
    group_list = [ii for ii in range(world_size)]
    new_pg = dist.new_group(group_list)
    
    model = SimpleNetwork().npu()
    tensor_in = torch.arange(world_size, dtype=torch.int64).npu() + (rank * world_size)
    tensor_out = torch.zeros(world_size, dtype=torch.int64).npu()
    expect_out = torch.zeros(world_size, dtype=torch.int64).npu()
    dist.all_to_all_single(expect_out, tensor_in, group=new_pg)

    compiled_model = torch.compile(
        model,
        backend=backend,
        fullgraph=True,
    )

    world_size = dist.get_world_size(new_pg)
    output = compiled_model(tensor_out, tensor_in, new_pg)

    check_group_info(new_pg)
    check_all_to_all_output(output, expect_out)
    dist.destroy_process_group()


if __name__ == "__main__":
    import sys
    test_name = sys.argv[1] if len(sys.argv) > 1 else None
    assert test_name is not None, f"test case name is None"
    exit_code = pytest.main([f"tests/st/inferrt/distributed/check_distributed_ops.py::{test_name}"])
    sys.exit(exit_code)