# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologies Co., Ltd.
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
"""mindspore distribution init"""

import argparse
import os

import mindspore as ms
from vllm.logger import init_logger, logger

logger = init_logger("vllm_mindspore.models")

def _get_host_and_ip(distributed_init_method):
    try:
        _, ip_str, port_str = distributed_init_method.split(":")
        ip = ip_str.split("/")[-1]
        port = int(port_str)
    except Exception as e:
        raise RuntimeError(
            "Cannot get host and port information from %s, error: %s!"
            % (distributed_init_method, str(e))
        )

    return ip, port


def _init_ms_distributed(rank_id, rank_size, distributed_init_method):
    comm_addr, comm_port = _get_host_and_ip(distributed_init_method)

    os.environ["MS_WORKER_NUM"] = str(rank_size)
    os.environ["MS_ROLE"] = "MS_SCHED"
    os.environ["MS_NODE_ID"] = str(rank_id)
    os.environ["MS_SCHED_HOST"] = str(comm_addr)
    os.environ["MS_SCHED_PORT"] = str(comm_port)
    os.environ["DEVICE_ID"] = str(rank_id)
    ms.communication.init()


if __name__ == "__main__":
    # TODO: need to be killed manually now, but supposed to be killed automatically
    parser = argparse.ArgumentParser()

    parser.add_argument("--rank_id", type=int, default=None, help="")
    parser.add_argument("--rank_size", type=int, default=None, help="")
    parser.add_argument("--distributed_init_method", type=str, default=None, help="")

    args = parser.parse_args()
    logger.warning(f"Run ms_scheduler_init.py")
    _init_ms_distributed(args.rank_id, args.rank_size, args.distributed_init_method)
    logger.warning(f"Complete run_scheduler_init.py")
