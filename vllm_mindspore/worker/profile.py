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

import os
import sys
import subprocess

from vllm.logger import init_logger

from mindspore import Profiler
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics
from mindspore.profiler.common.profiler_context import ProfilerContext

logger = init_logger(__name__)

PROFILE_ENV_NAME = "VLLM_TORCH_PROFILER_DIR"

def shell_analyse(path):
    subprocess.run(
        [sys.executable, "-c", f'from mindspore import Profiler; Profiler.offline_analyse("{path}")'],
        shell=False, check=True)

class AdapterProfiler:
    def __init__(self, path):
        self.profiler = Profiler(
            profiler_level=ProfilerLevel.Level1,
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            output_path=path,
            start_profile=False
        )

    def start(self):
        self.profiler.start()

    def stop(self):
        self.profiler.stop()
        path = ProfilerContext().ascend_ms_dir
        shell_analyse(path)

def wrapper_worker_init(fun):
    def new_fun(*arg, **kwarg):
        # Profiler initialization during worker init triggers device setup,
        # causing init_device to fail due to duplicate configuration.
        # To fix this, temporarily unset VLLM_TORCH_PROFILER_DIR before worker init,
        # restore it afterward, then initialize profiler properly after worker init_device completes
        profile_output_path = os.getenv(PROFILE_ENV_NAME, "")
        if profile_output_path:
            del os.environ[PROFILE_ENV_NAME]

        fun(*arg, **kwarg)

        if profile_output_path:
            os.environ[PROFILE_ENV_NAME] = profile_output_path
    return new_fun

def wrapper_worker_init_device(fun):
    def new_fun(*arg, **kwarg):
        fun(*arg, **kwarg)

        # The actual profiler initialization is performed after the worker.init_device() method,
        # based on the VLLM_TORCH_PROFILER_DIR environment variable.
        self = arg[0]
        profile_output_path = os.getenv(PROFILE_ENV_NAME, "")
        if profile_output_path:
            logger.info(f"Profiling enabled. Traces will be saved to: {profile_output_path}")
            self.profiler = AdapterProfiler(profile_output_path)
        else:
            self.profiler = None
    return new_fun


