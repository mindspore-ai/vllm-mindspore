# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/v1/worker/gpu_worker.py
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
"""Worker functions for ascend."""

import os
import subprocess
from typing import Any

import psutil
from vllm.logger import init_logger

from vllm_mindspore.utils import is_310p

logger = init_logger(__name__)


def compile_or_warm_up_model(self) -> None:
    # MindSpore does not support cuda graph. No need to warm up the model.
    # Since prefill is done previously, we do decode here.
    max_num_reqs = 1
    # Only pp_last_rank has lm_head, which is required by _dummy_sampler_run.

    # We skip EPLB here since we don't want to record dummy metrics
    hidden_states, last_hidden_states = \
        self.model_runner._dummy_run(
            num_tokens=max_num_reqs,
            skip_eplb=True,
        )
    if self.model_runner.is_pooling_model:
        self.model_runner._dummy_pooler_run(hidden_states)
    else:
        self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)


def execute_command(cmd_list):
    try:
        with subprocess.Popen(cmd_list,
                              shell=False,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) as p:
            out, _ = p.communicate(timeout=1000)
        res = out.decode()
        return res
    except FileNotFoundError as e:
        message = f"Failed to execute command, because {e}."
        raise RuntimeError(message) from e


def get_numa_map():
    npu_to_core_map = {}

    # Get quantity of CPUs and NUMA nodes.
    total_cpu_count = 0
    numa_node_count = 0
    numa_info = execute_command("lscpu").strip().split("\n")
    for val in numa_info:
        if val.startswith("CPU(s):"):
            total_cpu_count = int(val.split(" ")[-1])
        if val.startswith("NUMA"):
            numa_node_count = int(val.split(" ")[-1])
            break

    # Get chip count of NPU.
    chip_info = execute_command(["npu-smi", "info", "-l"]).strip().split("\n")
    chip_count = 0
    npu_count = 0
    for val in chip_info:
        if val.strip().startswith("Total"):
            npu_count = int(val.split(" ")[-1])
        if val.strip().startswith("Chip"):
            chip_count = int(val.split(" ")[-1])
            break

    # Get affinity relationship between CPUs and NPUs.
    numa_topo_info = execute_command(["npu-smi", "info", "-t",
                                      "topo"]).strip().split("\n")
    numa_to_npu_map: dict[Any, Any] = {}
    max_affinity_cpu = 0
    if "Affinity" not in numa_topo_info[0]:
        # If the device does not provide affinity,
        # the CPUs will be evenly distributed.
        cpu_num_per_npu = total_cpu_count // (npu_count * chip_count)
        for i in range(npu_count * chip_count):
            cpu_start = i * cpu_num_per_npu
            # 4 CPUs are reserved for CANN
            npu_to_core_map[i] = [cpu_start, cpu_start + cpu_num_per_npu - 4]
        return npu_to_core_map
    else:
        npu_num = 0
        for val in numa_topo_info[1:]:
            line = val.split(" ")
            if line and line[0].startswith("NPU"):
                cpu_affinity = line[-1]
                max_affinity_cpu = max(max_affinity_cpu,
                                       int(cpu_affinity.split("-")[1]))
                if numa_to_npu_map.get(cpu_affinity) is None:
                    numa_to_npu_map[cpu_affinity] = list()
                # If each NPU has multiple chips,
                # assign them to the same NUMA node.
                for i in range(chip_count):
                    numa_to_npu_map[cpu_affinity].append(npu_num * chip_count +
                                                         i)
                npu_num += 1

    # If the number of NUMA nodes with affinity is less than
    # or equal to half of the total, the offset is introduced,
    # and no extra CPU is reserved for CANN.
    if numa_node_count >= 2 * len(numa_to_npu_map):
        offset_mode = True
        cpu_reserved_for_cann = 0
    else:
        offset_mode = False
        cpu_reserved_for_cann = 4

    for key, val in numa_to_npu_map.items():
        cpu_range = key.split("-")
        cpu_start = int(cpu_range[0])
        cpu_end = int(cpu_range[1])
        cpu_count = cpu_end - cpu_start + 1
        if offset_mode:
            if max_affinity_cpu == total_cpu_count - 1:
                cpu_start = cpu_start - cpu_count
            else:
                cpu_start = cpu_start + cpu_count
        shared_npu_count = len(val)
        cpu_num_per_npu = int(cpu_count / shared_npu_count)
        for npu in val:
            npu_to_core_map[npu] = [
                cpu_start, cpu_start + cpu_num_per_npu - cpu_reserved_for_cann
            ]
            cpu_start += cpu_num_per_npu

    return npu_to_core_map


def bind_cpu(rank):
    rank_cpu_maps = get_numa_map()
    local_rank = rank % len(rank_cpu_maps)
    device_id = local_rank

    if "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
        device_control_env_var = os.environ["ASCEND_RT_VISIBLE_DEVICES"]
        device_id = int(device_control_env_var.split(",")[local_rank])

    cpu_range = rank_cpu_maps[device_id]
    cpu_list = list(range(cpu_range[0], cpu_range[1]))
    current_process = psutil.Process()
    current_process.cpu_affinity(cpu_list)
    logger.info("bind process %d in rank %d to cpu: %s", current_process.pid,
                local_rank, cpu_list)


def wrapper_worker_bind_cpu(fun):

    def new_fun(*arg, **kwargs):
        if not is_310p():
            # Bind CPU with wrapper when workers are initializing.
            local_rank = kwargs.get("local_rank")
            bind_cpu(local_rank)
        fun(*arg, **kwargs)

    return new_fun
