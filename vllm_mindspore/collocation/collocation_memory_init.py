# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2024 The vLLM team.
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
# ============================================================================

import argparse

import posix_ipc
from vllm.logger import init_logger

logger = init_logger("vllm.collocation.init")


def init_semaphore(name: str, initial_value: int):
    sem = posix_ipc.Semaphore(name=name,
                              flags=posix_ipc.O_CREAT,
                              initial_value=initial_value)
    while (sem.value > initial_value):
        sem.acquire()
    while (sem.value < initial_value):
        sem.release()
    logger.info("Semapore name=%s created with value=%s", sem.name, sem.value)
    return sem


def main(num_devices: int):
    logger.info("Init Collocation global shared memory")
    init_semaphore(name="/lock_prefill_launch", initial_value=1)
    for i in range(num_devices):
        init_semaphore(name=f"/lock_prefill_launch{i}", initial_value=1)
    init_semaphore(name="/lock_decode_launch", initial_value=1)
    for i in range(num_devices):
        init_semaphore(name=f"/lock_decode_launch{i}", initial_value=1)
    init_semaphore(name="/lock_logits_launch", initial_value=1)
    for i in range(num_devices):
        init_semaphore(name=f"/lock_logits_launch{i}", initial_value=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-devices",
                        type=int,
                        default=8,
                        help="Number of per-device semaphores to init")
    args = parser.parse_args()
    main(args.num_devices)
