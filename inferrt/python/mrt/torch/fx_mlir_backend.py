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

import torch

from mrt.torch.utils import (
    get_collective_info_from_torch,
    set_device_context,
    from_torch,
    to_torch,
    update_tensor_data,
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
    fx_importer.import_stateless_graph(gm.graph, func_name="main")
    torch_mlir_module = fx_importer.module
    return torch_mlir_module

def _parse_mlir_module_from_text(text: str):
    """Parse MLIR module from text IR."""
    # pylint: disable=import-outside-toplevel
    # Reason: mopt must be imported here to prevent default loading at module level
    from mopt import ir
    from mopt.dialects import torch as torch_d
    ctx = ir.Context()
    torch_d.register_dialect(ctx)
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

    _print_verbose("Original FX Graph", gm.graph)

    # Apply decompositions to FX GraphModule
    gm = apply_decompositions(gm, example_inputs)
    _print_verbose("FX Graph After Decompositions", gm.graph)

    # Convert FX GraphModule to torch_mlir RAW module
    torch_mlir_module = _convert_to_torch_mlir(gm)
    _print_verbose("Torch-MLIR RAW Module", torch_mlir_module)

    # Serialize torch_mlir module to IR text
    mlir_module = _parse_mlir_module_from_text(str(torch_mlir_module))
    _print_verbose("Re-parsed MLIR Module (torch_mlir RAW, before passes)", mlir_module)

    # Run pass pipeline to convert torch_mlir RAW to TORCH backend
    # pylint: disable=import-outside-toplevel
    # Reason: mopt must be imported here to prevent default loading at module level
    from mopt.passmanager import PassManager
    with mlir_module.context:
        pm = PassManager.parse("builtin.module(torchdynamo-export-to-torch-backend-pipeline)")
        pm.run(mlir_module.operation)
    _print_verbose("Torch Backend IR", mlir_module)

    # Run pass pipeline to convert torch_mlir TORCH backend to StableHLO
    with mlir_module.context:
        pm = PassManager.parse("builtin.module(torch-backend-to-stablehlo-backend-pipeline)")
        pm.run(mlir_module.operation)
    _print_verbose("StableHLO MLIR Module", mlir_module)

    # Run pass to convert StableHLO to MRT dialect
    with mlir_module.context:
        pm = PassManager.parse("builtin.module(convert-stablehlo-to-mrt)")
        pm.run(mlir_module.operation)
    _print_verbose("MRT Dialect Module (after passes)", mlir_module)

    # Convert RankedTensorType to Mrt_TensorType
    with mlir_module.context:
        pm = PassManager.parse("builtin.module(convert-ranked-tensor-to-mrt-tensor)")
        pm.run(mlir_module.operation)
    _print_verbose("MLIR Module (after type conversion)", mlir_module)

    # Extract device information from first example_input and run pass
    # Use PassOptions to pass device info - all logic is handled in C++ pass
    device_str = "cpu"
    device_index = -1
    if example_inputs and isinstance(example_inputs[0], torch.Tensor):
        device_str = _get_device_string_from_torch_tensor(example_inputs[0])
        device_index = getattr(example_inputs[0].device, "index", None)
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
    executor, placeholder_nodes = gen_executor_from_mlir_module(mlir_module)
    _print_verbose("Executor Graph", dump_func=executor.dump_graph)

    # Create compiled callable from GraphExecutor
    executor.build()
    return _create_compiled_callable(executor, placeholder_nodes)


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


def _create_compiled_callable(executor, placeholder_nodes):
    """Create a compiled callable from the executor."""
    def compiled_callable(*new_inputs: torch.Tensor):
        """Run compiled executor.

        Args:
            *new_inputs: New input tensors, count and types should match function parameters.

        Returns:
            Execution result (torch object).
            - If original function has single return value: returns single tensor
            - If original function has multiple return values: returns tuple
            - If original function has no return value: returns None or empty tuple

        Raises:
            ValueError: If input count mismatch.
        """
        if len(new_inputs) != len(placeholder_nodes):
            raise ValueError(
                f"Expected {len(placeholder_nodes)} inputs, "
                f"but received {len(new_inputs)}"
            )

        for i, p_node in enumerate(placeholder_nodes):
            if p_node.output.is_tensor():
                update_tensor_data(p_node.output.to_tensor(), new_inputs[i])
            else:
                p_node.output = from_torch(new_inputs[i])

        result = executor.run()
        return to_torch(result)

    return compiled_callable
