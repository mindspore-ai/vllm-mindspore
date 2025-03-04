#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import logging
import os

logger = logging.getLogger(__name__)


def env_setup(target_env_dict=None):
    if target_env_dict is None:
        target_env_dict = {
            "USE_TORCH": "FALSE",
            "USE_TF": "FALSE",
            "RUN_MODE": "prefict",
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
            "GLOG_v": "3"
        }

    for key, value in target_env_dict.items():
        if key not in os.environ:
            logger.debug('Setting %s to "%s"' % (key, value))
            os.environ[key] = value


def main():
    env_setup()

    from vllm.scripts import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
