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
import psutil


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
        '''
        env_var = os.environ.copy()
        env_var.update({
            "ASCEND_RT_VISIBLE_DEVICES": f"{device_str}",
            "LCAL_COMM_ID": f"{lccl_str}",
            "HCCL_IF_BASE_PORT": f"{hccl_base_port}"
        })

        out_tasks.append((env_var, task_case, log_file))

        '''
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

    if device_base > device_limit:
        raise ValueError(
            "Total require device %d exceeding resource limits %d !",
            device_base, device_limit)

    return out_tasks


def cleanup_subprocesses() -> None:
    """Cleanup all subprocesses raise by main test process."""
    cur_proc = psutil.Process(os.getpid())
    children = cur_proc.children(recursive=True)
    for child in children:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
