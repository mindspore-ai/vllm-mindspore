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

"""Configuration file for the 'lit' test runner."""

import os
import lit.formats
import lit.util
import lit.llvm

# pylint: disable=c-extension-no-member

lit.llvm.initialize(lit_config, config)

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "MRT"

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(True)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(lit_config.params["mrt_obj_root"], "tests")

# Tweak the PATH to include the tools dir.
# lit.llvm.llvm_config.with_environment('PATH', lit_config.params['mrt_tools_dir'], append_path=True)
# lit.llvm.llvm_config.with_environment('PATH', lit_config.params['llvm_tools_dir'], append_path=True)

# For each occurrence of a tool name, replace it with the full path to
# the build directory holding that tool.
mrt_tools = [
    "mrt-opt",
]
llvm_tools = [
    "FileCheck",
]

lit.llvm.llvm_config.add_tool_substitutions(
    mrt_tools, [lit_config.params["mrt_tools_dir"]]
)
lit.llvm.llvm_config.add_tool_substitutions(
    llvm_tools, [lit_config.params["llvm_tools_dir"]]
)
