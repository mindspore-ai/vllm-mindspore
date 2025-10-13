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

BACKEND_HCCL = "hccl"
def setup(rank, world_size):
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


def check_allgather_info(pg=None):
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

def check_allgather_output(output):
    output = output.cpu()
    expect_output = torch.tensor([ 100,  400,  400, 1600])
    assert (output == expect_output).all()

class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, gathered, pg=None):
        dist.all_gather_into_tensor(gathered, x, group=pg)
        output = torch.mul(gathered, gathered)
        return output


def train(rank, world_size):

    setup(rank, world_size)
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    new_pg = dist.new_group([0,1])
    
    model = SimpleNetwork().npu()
    example_input = torch.tensor([1,2]).npu() * (rank+1) * 10

    compiled_model = torch.compile(
        model,
        backend=backend,
        fullgraph=True,
    )

    world_size = dist.get_world_size(new_pg)
    print(f"rank {rank} world size {world_size}")
    gathered = [torch.zeros_like(example_input) for _ in range(world_size)]
    gathered = torch.cat(gathered, dim=0)
    output = compiled_model(example_input, gathered, new_pg)

    check_allgather_info(new_pg)
    check_allgather_output(output)

    print(f"rank {rank} world size {world_size} output is {output}")
    dist.destroy_process_group()

if __name__ == "__main__":
    rank = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    local_rank = int(os.getenv('LOCAL_RANK', '0'))

    torch.npu.set_device(local_rank)
    train(rank, world_size)
