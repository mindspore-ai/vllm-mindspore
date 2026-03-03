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

import logging
import os

# It's before the vllm import, so vllm.logger cannot be used here.
logger = logging.getLogger(__name__)


def _setup_batch_invariant_mode():
    """
    If VLLM_BATCH_INVARIANT is set to 1, enable deterministic mode
    """
    if os.environ.get("VLLM_BATCH_INVARIANT") == "1":
        logger.warning("Batch invariant is experimental feature related to "
                       "specific network implementations. Currently only "
                       "supports tensor parallel for DeepSeek models with "
                       "mindformers backend, and has no effect in other "
                       "scenarios. Note that VLLM_BATCH_INVARIANT environment "
                       "variable will enable the deterministic algorithm by "
                       "default.")

        batch_invariant_envs = {
            "LCCL_DETERMINISTIC": "1",
            "HCCL_DETERMINISTIC": "strict",
            "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
            "ATB_LLM_LCOC_ENABLE": "0"
        }

        for key, value in batch_invariant_envs.items():
            logger.debug('Setting %s to "%s"', key, value)
            os.environ[key] = value


def env_setup(target_env_dict=None):
    if target_env_dict is None:
        target_env_dict = {
            "USE_TORCH": "FALSE",
            "USE_TF": "FALSE",
            "RUN_MODE": "predict",
            "CUSTOM_MATMUL_SHUFFLE": "on",
            "HCCL_DETERMINISTIC": "false",
            "ASCEND_LAUNCH_BLOCKING": "0",
            "TE_PARALLEL_COMPILER": "0",
            "LCCL_DETERMINISTIC": "0",
            "MS_ENABLE_GRACEFUL_EXIT": "0",
            "CPU_AFFINIITY": "True",
            "MS_ENABLE_INTERNAL_BOOST": "on",
            "MS_ENABLE_LCCL": "off",
            "HCCL_EXEC_TIMEOUT": "7200",
            "DEVICE_NUM_PER_NODE": "16",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "MS_JIT_MODULES": "vllm_mindspore,research",
            "GLOG_v": "3",
            "RAY_CGRAPH_get_timeout": "360",
            # For CPU communication timeout setting,
            # default is 15s, change to 180s
            # to avoid multi node timeout when starting service.
            "MS_NODE_TIMEOUT": "180",
            # aclgraph current support capture 19 graphs as
            # the total subgraph number is 2000
            "MS_DEV_RUNTIME_CONF": "graph_capture_max_number:19",
            "MS_ALLOC_CONF": "enable_vmm: true, derag_memory_step_freq: -1"
        }

    for key, value in target_env_dict.items():
        if key not in os.environ:
            logger.debug('Setting %s to "%s"', key, value)
            os.environ[key] = value

    _setup_batch_invariant_mode()


def main():
    env_setup()

    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
