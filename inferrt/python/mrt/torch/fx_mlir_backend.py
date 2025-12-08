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

"""FX-to-MLIR backend utilities.

This module provides:
- An FX backend entry that applies decompositions and imports to StableHLO.
- Integration with MRT dialect through the _executor_builder module.
"""

import os
from typing import List
from collections import namedtuple

import torch

from mrt.torch.utils import (
    get_collective_info_from_torch,
    set_device_context,
    to_torch,
    update_runtime_inputs,
)
from mrt.torch._decompositions import apply_decompositions
from mrt.torch._executor_builder import gen_executor_from_mlir_module

# print IR flag
_PRINT_IR = os.environ.get("MOPT_PRINT_IR") == "1"


def _print_verbose(stage_title: str, content=None, dump_func=None):
    """Print verbose information with formatting.

    Args:
        stage_title: Title of the stage
        content: Content to print (if provided)
        dump_func: Optional callable that prints to stdout (if provided, called after printing content)
    """
    if not _PRINT_IR:
        return
    print("=" * 80)
    print(stage_title)
    print("=" * 80)
    if content is not None:
        print(content)
    if dump_func is not None:
        dump_func()
    print()


def _get_var_to_range(gm: torch.fx.GraphModule):
    """Get variable to range mapping from FX GraphModule.

    Args:
        gm: FX GraphModule

    Returns:
        A dictionary mapping variable names to their corresponding range constraints.
    """
    shape_env = None
    for node in gm.graph.nodes:
        if "val" in node.meta:
            val = node.meta["val"]
            if hasattr(val, "fake_mode") and val.fake_mode is not None:
                shape_env = val.fake_mode.shape_env
                if shape_env is not None:
                    break

    if shape_env is None or not hasattr(shape_env, "var_to_range"):
        return {}
    return shape_env.var_to_range


def _convert_to_torch_mlir(gm: torch.fx.GraphModule):
    """Convert FX GraphModule to torch_mlir RAW module."""
    # pylint: disable=import-outside-toplevel
    # Reason: torch_mlir must be imported here to prevent default loading at module level
    from torch_mlir import ir
    from torch_mlir.extras.fx_importer import FxImporter
    from torch_mlir.dialects import torch as torch_d

    torch_mlir_context = ir.Context()
    torch_d.register_dialect(torch_mlir_context)
    fx_importer = FxImporter(context=torch_mlir_context)

    # Mocking a ExportedProgram to make FxImporter work properly on symbolic shape
    FakeExportedProgram = namedtuple("FakeExportedProgram", ["range_constraints"])
    # pylint: disable=protected-access
    fx_importer._cc.set_symbolic_guards(FakeExportedProgram(_get_var_to_range(gm)))

    fx_importer.import_stateless_graph(
        gm.graph,
        func_name="main",
        import_symbolic_shape_expressions=True,
    )
    torch_mlir_module = fx_importer.module
    return torch_mlir_module


def _parse_mlir_module_from_text(text: str):
    """Parse MLIR module from text IR."""
    # pylint: disable=import-outside-toplevel
    # Reason: mopt must be imported here to prevent default loading at module level
    from mopt import ir
    from mopt.dialects import torch as torch_d
    from mopt import register_mrt_dialect

    ctx = ir.Context()
    torch_d.register_dialect(ctx)
    register_mrt_dialect(ctx)
    return ir.Module.parse(text, ctx)


def backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """FX backend entry point: FX GraphModule -> StableHLO -> MRT dialect -> GraphExecutor.

    Args:
        gm: torch.fx.GraphModule instance
        example_inputs: Example input list for type inference

    Returns:
        Callable executor function

    Environment Variables:
        MOPT_PRINT_IR: Set to 1 to print IR at each stage
    """
    get_collective_info_from_torch(gm)
    set_device_context()

    _print_verbose("Original FX Graph", dump_func=gm.print_readable)

    # Apply decompositions to FX GraphModule
    fake_inputs = [
        node.meta["example_value"]
        for node in gm.graph.nodes
        if node.op == "placeholder"
    ]
    m = apply_decompositions(gm, fake_inputs)
    _print_verbose("FX Graph After Decompositions", dump_func=m.print_readable)

    # Convert FX GraphModule to torch_mlir RAW module
    torch_mlir_module = _convert_to_torch_mlir(m)
    _print_verbose("Torch-MLIR RAW Module", torch_mlir_module)

    # Serialize torch_mlir module to IR text
    mlir_module = _parse_mlir_module_from_text(str(torch_mlir_module))
    _print_verbose("Re-parsed MLIR Module (torch_mlir RAW, before passes)", mlir_module)

    # Run pass pipeline to convert torch_mlir RAW to TORCH backend
    # pylint: disable=import-outside-toplevel
    # Reason: mopt must be imported here to prevent default loading at module level
    from mopt.passmanager import PassManager

    with mlir_module.context:
        pm = PassManager.parse(
            "builtin.module(torchdynamo-export-to-torch-backend-pipeline)"
        )
        pm.run(mlir_module.operation)
    _print_verbose("Torch Backend IR", mlir_module)

    # Run pass to convert Torch to MRT dialect directly
    with mlir_module.context:
        pm = PassManager.parse("builtin.module(convert-torch-to-mrt,canonicalize)")
        pm.run(mlir_module.operation)
    _print_verbose("MRT Dialect Module (after passes)", mlir_module)

    # Extract device information from first example_input and run pass
    # Use PassOptions to pass device info - all logic is handled in C++ pass
    device_str = "cpu"
    device_index = -1
    if example_inputs:
        first_tensor = next((inp for inp in example_inputs if isinstance(inp, torch.Tensor)), None)
        if first_tensor is not None:
            device_str = _get_device_string_from_torch_tensor(first_tensor)
            device_index = getattr(first_tensor.device, "index", None)
            if device_index is None:
                device_index = -1

    # Run pass with device information via PassOptions
    with mlir_module.context:
        # Pass device info through PassManager.parse() options
        pm = PassManager.parse(
            f"builtin.module(set-tensor-device{{device-type={device_str} device-index={device_index}}})"
        )
        pm.run(mlir_module.operation)
    _print_verbose("MRT Dialect Module (after set-tensor-device pass)", mlir_module)

    # Build MRT dialect module into GraphExecutor
    executor, param_nodes = gen_executor_from_mlir_module(mlir_module, fake_inputs)
    _print_verbose("Executor Graph", dump_func=executor.dump_graph)

    # Create compiled callable from GraphExecutor
    executor.build()

    def compiled_callable(*inputs: torch.Tensor):
        update_runtime_inputs(param_nodes, inputs)
        result = executor.run()
        return to_torch(result)

    return compiled_callable


def _get_device_string_from_torch_tensor(tensor: torch.Tensor) -> str:
    """Extract device type string from torch tensor.

    Args:
        tensor: PyTorch tensor

    Returns:
        Device type string: "cpu" or "npu"
    """
    device = tensor.device
    device_type_str = str(device.type)
    if device_type_str == "cpu":
        return "cpu"
    if device_type_str in ("npu", "privateuse1"):
        # torch.device('npu') uses PrivateUse1 device type
        return "npu"
    # Default to cpu for unknown device types
    return "cpu"
