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

import psutil


class EnvVarManager:

    @staticmethod
    def setup_mindformers_environment():
        """Set MindFormers to PYTHONPATH"""
        # Insert mindformers to PYTHONPATH.
        mindformers_path =\
            "/home/jenkins/mindspore/testcases/testcases/tests/mindformers"

        if mindformers_path not in sys.path:
            sys.path.insert(0, mindformers_path)

        current_pythonpath = os.environ.get("PYTHONPATH", "")
        if current_pythonpath:
            if mindformers_path not in current_pythonpath:
                os.environ[
                    "PYTHONPATH"] = f"{mindformers_path}:{current_pythonpath}"
        else:
            os.environ["PYTHONPATH"] = mindformers_path
