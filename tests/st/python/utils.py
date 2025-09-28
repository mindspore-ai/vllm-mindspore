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

import contextlib
import os
import signal
import sys
from typing import Optional

import psutil


class EnvVarManager:

    def __init__(self):
        self._original_env: dict[str, Optional[str]] = {}
        self._managed_vars: dict[str, str] = {}

    def set_env_var(self, var_name: str, value: str) -> None:
        """Set environment variable and record original value."""
        # Record original values corresponding to var_name, None if not exist.
        if var_name not in self._original_env:
            self._original_env[var_name] = os.environ.get(var_name)

        os.environ[var_name] = value
        self._managed_vars[var_name] = value

    def unset_env_var(self, var_name: str) -> None:
        """Unset environment variable with original value."""
        if var_name not in self._original_env:
            raise ValueError(
                f"Variable {var_name} was not set by this manager")

        original_value = self._original_env[var_name]
        if original_value is not None:
            os.environ[var_name] = original_value
        else:
            if var_name in os.environ:
                del os.environ[var_name]

        del self._original_env[var_name]
        del self._managed_vars[var_name]

    def unset_all(self) -> None:
        """Unset all environment variables with original values."""
        for var_name in list(self._managed_vars.keys()):
            self.unset_env_var(var_name)

    def get_managed_vars(self) -> dict[str, str]:
        """get all managered variables."""
        return self._managed_vars.copy()

    def setup_ai_environment(self, env_vars: dict[str, str]) -> None:
        """Set ai environment by given values."""
        # Insert mindformers to PYTHONPATH.
        mindformers_path =\
            "/home/jenkins/mindspore/testcases/testcases/tests/mindformers"

        if mindformers_path not in sys.path:
            sys.path.insert(0, mindformers_path)

        current_pythonpath = os.environ.get("PYTHONPATH", "")
        if current_pythonpath:
            os.environ[
                "PYTHONPATH"] = f"{mindformers_path}:{current_pythonpath}"
        else:
            os.environ["PYTHONPATH"] = mindformers_path

        os.environ['MS_ENABLE_TRACE_MEMORY'] = "off"

        # Update environments.
        for var_name, value in env_vars.items():
            self.set_env_var(var_name, value)


def cleanup_subprocesses() -> None:
    """Cleanup all subprocesses raise by main test process."""
    cur_proc = psutil.Process(os.getpid())
    children = cur_proc.children(recursive=True)
    for child in children:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)


def tasks_resource_alloc(tasks: list[tuple[int]]) -> list[tuple[str]]:
    """
    Allocate devices, lccl base port, hccl base port to tasks
    according to device requirement of each task.

    For example:
        [(2, "cases_parallel/vllm_task.py::test_1", "test_1.log")]
        ==> [("export ASCEND_RT_VISIBLE_DEVICES=0,1 &&
               export LCAL_COMM_ID=127.0.0.1:10068 && "
              "export HCCL_IF_BASE_PORT=61000 && "
              "pytest -s -v cases_parallel/vllm_task.py::test_1 > test_1.log",
              "test_1.log")]

    Args:
        tasks (list[tuple[int]]): list of tasks. Each task contain 3 elements.
            1. device_req (int): Num of device requirements,
                                 which will occur device_req devices,
                                 device_req ports for lccl,
                                 device_req ports for hccl.
            2. case_desc (str): The case description,
               such as "path_to_case/case.py::target_case".
            3. log_file (str): The logging file path.

    Returns:
        list[tuple[str]]: Append resource environment to the task commands.
    """
    device_limit = 8
    device_base = 0
    lccl_base_port = 10068
    hccl_base_port = 61000

    out_tasks: list[tuple[str]] = []
    for task in tasks:
        assert len(task) == 3
        resource_req, task_case, log_file = task
        if not isinstance(resource_req, int):
            raise TypeError(
                "First argument of task should be a int or str, but got %s!",
                str(type(resource_req)))

        device_str = ",".join(
            [str(d) for d in range(device_base, device_base + resource_req)])
        lccl_str = f"127.0.0.1:{lccl_base_port}"

        commands = [
            f"export ASCEND_RT_VISIBLE_DEVICES={device_str}",
            f"export LCAL_COMM_ID={lccl_str}",
            f"export HCCL_IF_BASE_PORT={hccl_base_port}"
        ]
        device_base += resource_req
        lccl_base_port += resource_req
        hccl_base_port += resource_req

        commands.append(f"pytest -s -v {task_case} > {log_file}")
        out_tasks.append((" && ".join(commands), log_file))

    if device_limit > device_limit:
        raise ValueError(
            "Total require device %d exceeding resource limits %d !",
            device_base, device_limit)

    return out_tasks
