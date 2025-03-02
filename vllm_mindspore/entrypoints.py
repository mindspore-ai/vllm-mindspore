#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
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

import importlib
import inspect
import subprocess
import sys
import tempfile
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Not enough arguments for entrypoint!")

    module_name = sys.argv[1]
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        raise ValueError(
            "Invalid entrypoint(%s) for vllm, error: %s!" % (module_name, str(e))
        )

    module_code = inspect.getsource(module)
    vllm_mindspore_enable_line = "import vllm_mindspore\n"
    module_code = vllm_mindspore_enable_line + module_code

    with tempfile.TemporaryDirectory() as temp_folder:
        exec_file = Path(temp_folder) / "temp_entrypoint.py"
        with open(exec_file, "w") as f:
            f.writelines(module_code)

        subprocess.run([sys.executable, exec_file] + sys.argv[2:])
