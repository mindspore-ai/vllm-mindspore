# SPDX-License-Identifier: Apache-2.0

# Functions are adapted from vllm-project/vllm/v1/executor/multiproc_executor.py
#
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
"""Monkey Patch functions for v1 executor mp distributed backend."""
import os
import signal
import time

from vllm.logger import init_logger

logger = init_logger(__name__)


def executor_ensure_worker_termination(self):
    """Ensure that all worker processes are terminated. Assumes workers have
    received termination requests. Waits for processing, then sends
    termination and kill signals if needed."""

    def wait_for_termination(procs, timeout):
        if not time:
            # If we are in late stage shutdown, the interpreter may replace
            # `time` with `None`.
            return all(not proc.is_alive() for proc in procs)
        start_time = time.time()
        while time.time() - start_time < timeout:
            if all(not proc.is_alive() for proc in procs):
                return True
            time.sleep(0.1)
        return False

    # Send SIGTERM if still running
    active_procs = [w.proc for w in self.workers if w.proc.is_alive()]
    for p in active_procs:
        p.terminate()
    if not wait_for_termination(active_procs, 4):
        # Send SIGKILL if still running
        active_procs = [p for p in active_procs if p.is_alive()]
        for p in active_procs:
            # vllm-mindspore begin: kill all the process in the process group
            # (including scheduler process, kernel process and so on) instead of
            # calling p.kill.
            pid = p.pid
            try:
                os.killpg(pid, signal.SIGKILL)
            except Exception as e:
                logger.debug("Kill process %d error: %s!", pid, str(e))
            # vllm-mindspore end.

    self._cleanup_sockets()
