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

logger = init_logger("vllm.collocation.clean")


def clean_semaphore(name: str):
    try:
        posix_ipc.unlink_semaphore(name)
        logger.info("Semaphore %s unlink", name)
    except posix_ipc.ExistentialError:
        logger.info("Semaphore %s does not exist", name)


def main(num_devices: int):
    logger.info("Clean Collocation global shared memory")
    clean_semaphore(name="/lock_prefill_launch")
    for i in range(num_devices):
        clean_semaphore(name=f"/lock_prefill_launch{i}")
    clean_semaphore(name="/lock_decode_launch")
    for i in range(num_devices):
        clean_semaphore(name=f"/lock_decode_launch{i}")
    clean_semaphore(name="/lock_logits_launch")
    for i in range(num_devices):
        clean_semaphore(name=f"/lock_logits_launch{i}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-devices",
                        type=int,
                        default=8,
                        help="Number of per-device semaphores to init")
    args = parser.parse_args()
    main(args.num_devices)
