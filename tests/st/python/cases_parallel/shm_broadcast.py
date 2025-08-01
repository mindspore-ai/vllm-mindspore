# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/tests/distributed/test_shm_broadcast.py
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
"""test cpu communicator and share memory"""

# type: ignore
# isort: skip_file

import multiprocessing
import random
import time
from typing import List

import numpy as np
import torch.distributed as dist

import vllm_mindspore

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.utils import get_ip, get_open_port, get_distributed_init_method
from tests.st.python import utils


def teardown_function():
    utils.cleanup_subprocesses()


def get_arrays(n: int, seed: int = 0) -> List[np.ndarray]:
    np.random.seed(seed)
    sizes = np.random.randint(1, 10_000, n)
    # on average, each array will have 5k elements
    # with int64, each array will have 40kb
    return [np.random.randint(1, 100, i) for i in sizes]


def distributed_run(fn, world_size):
    number_of_processes = world_size
    processes = []

    port = get_open_port()
    distributed_init_method = get_distributed_init_method("127.0.0.1", port)

    for i in range(number_of_processes):
        p = multiprocessing.Process(target=fn,
                                    args=(distributed_init_method, i,
                                          world_size))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for p in processes:
        assert p.exitcode == 0


def worker_fn_wrapper(fn):
    # `multiprocessing.Process` cannot accept environment variables directly
    # so we need to pass the environment variables as arguments
    # and update the environment variables in the function
    def wrapped_fn(distributed_init_method, rank, world_size):
        dist.init_process_group(
            backend="nccl",
            init_method=distributed_init_method,
            rank=rank,
            world_size=world_size,
        )
        fn()

    return wrapped_fn


@worker_fn_wrapper
def worker_fn():
    rank = dist.get_rank()
    if rank == 0:
        port = get_open_port()
        ip = get_ip()
        dist.broadcast_object_list([ip, port], src=0)
    else:
        recv = [None, None]
        dist.broadcast_object_list(recv, src=0)
        ip, port = recv

    stateless_pg = dist.new_group([0, 1, 2, 3], backend="gloo")

    for pg in [dist.group.WORLD, stateless_pg]:

        writer_rank = 2
        broadcaster = MessageQueue.create_from_process_group(
            pg, 40 * 1024, 2, writer_rank)
        if rank == writer_rank:
            seed = random.randint(0, 1000)
            dist.broadcast_object_list([seed], writer_rank)
        else:
            recv = [None]
            dist.broadcast_object_list(recv, writer_rank)
            seed = recv[0]  # type: ignore

        if pg == dist.group.WORLD:
            dist.barrier()
        else:
            dist.barrier(group=pg)

        # in case we find a race condition
        # print the seed so that we can reproduce the error
        print(f"Rank {rank} got seed {seed}")
        # test broadcasting with about 400MB of data
        N = 10_000
        if rank == writer_rank:
            arrs = get_arrays(N, seed)
            for x in arrs:
                broadcaster.broadcast_object(x)
                time.sleep(random.random() / 1000)
        else:
            arrs = get_arrays(N, seed)
            for x in arrs:
                y = broadcaster.broadcast_object(None)
                assert np.array_equal(x, y)
                time.sleep(random.random() / 1000)

        if pg == dist.group.WORLD:
            dist.barrier()
            print("torch distributed passed the test!")
        else:
            dist.barrier(group=pg)
            print("StatelessProcessGroup passed the test!")


def test_shm_broadcast():
    distributed_run(worker_fn, 4)
