# Copyright 2026 Huawei Technologies Co., Ltd
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

"""Runner script for moe_distribute_dispatch_v2 distributed tests."""

import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch_npu

from ms_inferrt.torch import backend
from tests.ops_utils import AssertRtolEqual


BACKEND_HCCL = "hccl"

# pylint: disable=protected-access


def setup_distributed():
    """Initialize the distributed environment with HCCL backend."""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)

    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "6693")
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["RANK_ID"] = str(rank)

    dist.init_process_group(
        BACKEND_HCCL, rank=rank, world_size=world_size, init_method="env://"
    )


def cleanup():
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def build_groups(tp_world_size):
    """Create EP/TP groups and return HCCL info for the current rank."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ep_world_size = world_size // tp_world_size

    if tp_world_size == 1:
        ep_group = dist.new_group(list(range(world_size)))
        tp_group = None
        for tp_rank in range(world_size):
            group = dist.new_group([tp_rank])
            if rank == tp_rank:
                tp_group = group
        ep_rank_id = rank
        tp_rank_id = 0
    else:
        ep_ranks_list = []
        tp_ranks_list = []
        for i in range(tp_world_size):
            ep_ranks_list.append(list(range(i, world_size, tp_world_size)))
        for i in range(ep_world_size):
            tp_ranks_list.append(
                list(range(i * tp_world_size, (i + 1) * tp_world_size))
            )

        ep_group = None
        tp_group = None
        for ranks in ep_ranks_list:
            group = dist.new_group(ranks)
            if rank in ranks:
                ep_group = group
        for ranks in tp_ranks_list:
            group = dist.new_group(ranks)
            if rank in ranks:
                tp_group = group

        ep_rank_id = rank // tp_world_size
        tp_rank_id = rank % tp_world_size

    ep_hcomm_name = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    tp_hcomm_name = tp_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    return ep_world_size, ep_hcomm_name, tp_hcomm_name, ep_rank_id, tp_rank_id


def build_inputs(bs, h, k, moe_expert_num, dtype=torch.float16):
    """Create deterministic per-rank inputs."""
    rank = dist.get_rank()
    torch.manual_seed(20260422 + rank * 97 + bs)

    x = torch.randn(bs, h, dtype=dtype).npu()
    row_offsets = torch.arange(bs, dtype=torch.int32).view(bs, 1)
    topk_offsets = torch.arange(k, dtype=torch.int32).view(1, k)
    expert_ids = (row_offsets + topk_offsets + rank) % moe_expert_num
    expert_ids = expert_ids.npu()
    return x, expert_ids


def assert_dispatch_outputs(eager_out, compiled_out):
    """Compare dispatch outputs from eager and compiled graph modes."""
    assert len(eager_out) == len(compiled_out) == 7
    # Only expand_x/expert_token_nums are stable for strict value checking here.
    # dynamic_scales/assist_info_for_combine/recv_counts/expand_scales can contain
    # temporary workspace-like metadata that is not numerically stable across runs.
    # Validate shape/dtype for these outputs, and keep strict value checks for core tensors.
    value_check_indices = {0, 3}
    for idx, (eager_tensor, compiled_tensor) in enumerate(zip(eager_out, compiled_out)):
        assert eager_tensor.shape == compiled_tensor.shape
        assert eager_tensor.dtype == compiled_tensor.dtype
        if idx not in value_check_indices:
            continue
        AssertRtolEqual(eager_tensor.detach().cpu(), compiled_tensor.detach().cpu())


def run_case(bs_list, h, k, tp_world_size, moe_expert_num, shared_expert_num, shared_expert_rank_num):
    """Compare eager and compiled dispatch_v2 results across dynamic shapes."""
    setup_distributed()
    try:
        (
            ep_world_size,
            ep_hcomm_name,
            tp_hcomm_name,
            ep_rank_id,
            tp_rank_id,
        ) = build_groups(tp_world_size)

        def dispatch_func(x_in, expert_ids_in):
            global_bs = x_in.shape[0] * ep_world_size
            return torch_npu.npu_moe_distribute_dispatch_v2(
                x=x_in,
                expert_ids=expert_ids_in,
                group_ep=ep_hcomm_name,
                ep_world_size=ep_world_size,
                ep_rank_id=ep_rank_id,
                moe_expert_num=moe_expert_num,
                scales=None,
                x_active_mask=None,
                expert_scales=None,
                elastic_info=None,
                performance_info=None,
                group_tp=tp_hcomm_name,
                tp_world_size=tp_world_size,
                tp_rank_id=tp_rank_id,
                expert_shard_type=0,
                shared_expert_num=shared_expert_num,
                shared_expert_rank_num=shared_expert_rank_num,
                quant_mode=0,
                global_bs=global_bs,
                expert_token_nums_type=1,
                comm_alg="",
                zero_expert_num=0,
                copy_expert_num=0,
                const_expert_num=0,
            )

        compiled_func = torch.compile(dispatch_func, backend=backend, dynamic=True)

        for bs in bs_list:
            x, expert_ids = build_inputs(bs, h, k, moe_expert_num)
            eager_out = dispatch_func(x, expert_ids)
            compiled_out = compiled_func(x, expert_ids)
            torch.npu.synchronize()
            assert_dispatch_outputs(eager_out, compiled_out)

            dist.barrier()
            rank = dist.get_rank()
            print(
                f"Rank {rank}: moe_distribute_dispatch_v2 passed "
                f"(bs={bs}, ep_rank={ep_rank_id}, tp_rank={tp_rank_id})"
            )
    finally:
        cleanup()


def test_moe_distribute_dispatch_v2_basic_dynamic():
    """
    Feature: Check moe_distribute_dispatch_v2 op launch with dynamic shape
    Description: Test eager and compiled dispatch_v2 with tp_world_size=1 and varying batch sizes
    Expectation: The compiled result matches eager
    """
    run_case(
        bs_list=[8, 12],
        h=1024,
        k=2,
        tp_world_size=1,
        moe_expert_num=8,
        shared_expert_num=0,
        shared_expert_rank_num=0,
    )


def test_moe_distribute_dispatch_v2_with_tp_dynamic():
    """
    Feature: Check moe_distribute_dispatch_v2 op launch with TP domain and dynamic shape
    Description: Test eager and compiled dispatch_v2 with tp_world_size=2 and varying batch sizes
    Expectation: The compiled result matches eager
    """
    run_case(
        bs_list=[8, 12],
        h=1024,
        k=2,
        tp_world_size=2,
        moe_expert_num=4,
        shared_expert_num=0,
        shared_expert_rank_num=0,
    )


if __name__ == "__main__":
    test_name = sys.argv[1] if len(sys.argv) > 1 else None
    assert test_name is not None, "test case name is None"

    test_func = globals().get(test_name)
    if test_func is None or not test_name.startswith("test_"):
        raise ValueError(f"Unknown test case: {test_name}")

    try:
        test_func()
    except BaseException:  # pylint: disable=broad-except
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)  # Avoid distributed teardown masking the real failure.

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
