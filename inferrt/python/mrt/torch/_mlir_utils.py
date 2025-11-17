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

"""MLIR utility functions module.

This module provides:
- Conversion passes from StableHLO to MRT dialect
- MLIR module serialization and parsing
"""

# pylint: disable=import-outside-toplevel
# Reason: These imports are expensive and only needed when the functions are called


def run_stablehlo_to_mrt_passes(mlir_module) -> None:
    """Convert StableHLO module to MRT dialect through pass pipeline."""
    from mopt.passmanager import PassManager
    with mlir_module.context:
        pm = PassManager.parse("builtin.module(convert-stablehlo-to-mrt)")
        pm.run(mlir_module.operation)


def mlir_module_to_text(mlir_module) -> str:
    """Serialize MLIR module to text IR."""
    return str(mlir_module)


def parse_mlir_module_from_text(text: str):
    """Parse MLIR module from text IR."""
    from mopt import ir
    ctx = ir.Context()
    return ir.Module.parse(text, ctx)
