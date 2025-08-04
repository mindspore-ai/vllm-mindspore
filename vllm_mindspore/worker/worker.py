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
"""Adapted functions for mindspore in Worker."""

import math
import os
import subprocess

import psutil
import torch
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceGroupMetadata

from vllm_mindspore.utils import get_valid_dtype, is_310p

logger = init_logger(__name__)


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
    numa_to_npu_map = {}
    max_affinity_cpu = 0
    if "Affinity" not in numa_topo_info[0] or is_310p():
        # If the device does not provide affinity,
        # the CPUs will be evenly distributed.
        cpu_num_per_npu = total_cpu_count // (npu_count * chip_count)
        for i in range(npu_count * chip_count):
            cpu_start = i * cpu_num_per_npu
            # 4 CPUs are reserved for CANN(not for 310p)
            npu_to_core_map[i] = [
                cpu_start,
                cpu_start + cpu_num_per_npu - (0 if is_310p() else 4)
            ]
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
        # Bind CPU with wrapper when workers are initializing.
        # Support 910B, 910C and 310P.
        local_rank = kwargs.get("local_rank")
        parallel_config = kwargs.get("vllm_config").parallel_config
        local_rank = (parallel_config.data_parallel_rank_local *
                      parallel_config.world_size + local_rank)
        bind_cpu(local_rank)
        fun(*arg, **kwargs)

    return new_fun


def _prepare_input_for_warmup(model_config,
                              model_runner,
                              cache_engine,
                              is_prefill,
                              is_mtp_model=False):
    bs = 1
    seq_len = model_runner.scheduler_config.max_num_batched_tokens \
        if is_prefill else 1
    dummy_data = model_runner.input_registry.dummy_data_for_profiling(
        model_config, seq_len, model_runner.mm_registry)
    block_tables_num = [
        i for i in range(math.ceil(seq_len / cache_engine.block_size))
    ]

    # adapter multi modal warm up
    seq_data = dummy_data.seq_data
    if seq_len == 1:
        seq_data = dummy_data.seq_data.from_prompt_token_counts((0, seq_len))

    seqs = [
        SequenceGroupMetadata(
            request_id=str(idx),
            is_prompt=is_prefill,
            seq_data={idx: seq_data},
            sampling_params=SamplingParams(),
            block_tables={idx: block_tables_num},
            lora_request=None,
            multi_modal_data=None,
            multi_modal_placeholders=None,
        ) for idx in range(bs)
    ]

    model_input = model_runner.prepare_model_input(seqs)
    previous_hidden_states = None if not is_mtp_model else torch.ones(
        [bs, seq_len, model_config.get_hidden_size()],
        dtype=get_valid_dtype(model_config.dtype))
    return model_input, previous_hidden_states


def _warm_up_model(self) -> None:
    # cache_engine is a list with length equal to the size of
    # pipeline-parallel, and only pp=1 is supported.
    kv_cache = self.cache_engine[0].gpu_cache
    is_mtp_model = self.speculative_config is not None and \
        self.model_config.hf_config.model_type == "deepseek_mtp"

    def get_model(cls):
        if cls.vllm_config.scheduler_config.is_multi_step:
            return cls.model_runner._base_model_runner.model
        return cls.model_runner.model

    intermediate_tensors = None
    model = get_model(self)
    if not get_pp_group().is_first_rank:
        intermediate_tensors = model.make_empty_intermediate_tensors(
            batch_size=1,
            dtype=self.model_config.dtype,
            device=self.devices,
        )
    if is_mtp_model:
        # prefill mtp model
        model_input, previous_hidden_states = _prepare_input_for_warmup(
            self.model_config, self.model_runner, self.cache_engine[0], True,
            is_mtp_model)
        self.model_runner.execute_model(
            model_input,
            kv_cache,
            intermediate_tensors,
            previous_hidden_states=previous_hidden_states)

    # warmup for decode
    if self.vllm_config.scheduler_config.is_multi_step:
        model_input, _ = _prepare_input_for_warmup(
            self.model_config, self.model_runner._base_model_runner,
            self.cache_engine[0], False)
        self.model_runner._base_model_runner.execute_model(
            model_input, kv_cache, intermediate_tensors)
    else:
        model_input, previous_hidden_states = _prepare_input_for_warmup(
            self.model_config, self.model_runner, self.cache_engine[0], False,
            is_mtp_model)
        self.model_runner.execute_model(
            model_input,
            kv_cache,
            intermediate_tensors,
            previous_hidden_states=previous_hidden_states)

    torch.cuda.synchronize()

    # Reset the seed to ensure that the random state is not affected by
    # the model initialization and profiling.
    set_random_seed(self.model_config.seed)
