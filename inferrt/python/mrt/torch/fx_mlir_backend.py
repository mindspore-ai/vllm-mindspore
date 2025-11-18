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
from typing import List, Mapping, Optional

import torch
from torch._decomp import get_decompositions
from torch.fx.experimental.proxy_tensor import make_fx
from torch.func import functionalize

from mrt.torch.utils import (
    get_collective_info_from_torch,
    set_device_context,
    from_torch,
    to_torch,
    update_tensor_data,
)
from mrt.torch._mlir_utils import (
    run_stablehlo_to_mrt_passes,
    mlir_module_to_text,
    parse_mlir_module_from_text,
)
from mrt.torch._decompositions import DEFAULT_DECOMPOSITIONS
from mrt.torch._executor_builder import gen_executor_from_mlir_module


def apply_decompositions(
        gm: torch.fx.GraphModule,
        example_inputs,
        decompose_ops: Optional[List[torch._ops.OpOverload]] = None,  # pylint: disable=protected-access
):
    """Apply operator decompositions to a GraphModule if requested.

    Note:
        The torch._ops types are part of PyTorch's operator registry surface.
        We suppress Pylint's protected-access warning for the type annotation.
    """
    if decompose_ops is None:
        return gm

    decompositions: Mapping = get_decompositions(decompose_ops)
    gm = make_fx(
        functionalize(gm),
        decomposition_table=decompositions,
    )(*example_inputs)

    return gm


def backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """FX backend entry point: FX GraphModule -> StableHLO -> MRT dialect -> GraphExecutor.

    Workflow:
    1. Use torch_mlir.fx.stateless_fx_import to import FX GraphModule as StableHLO module
    2. Serialize StableHLO module to text
    3. Re-parse from text to MLIR module (reserved optimization interface)
    4. Run pass to convert StableHLO to MRT dialect
    5. Build MRT dialect module into GraphExecutor

    Args:
        gm: torch.fx.GraphModule instance
        example_inputs: Example input list for type inference

    Returns:
        Callable executor function

    Environment Variables:
        MOPT_PRINT_IR: Set to 1 to print IR at each stage
    """
    set_device_context()

    # pylint: disable=import-outside-toplevel
    # Reason: torch_mlir must be imported here to prevent default loading at module level
    from torch_mlir import fx  # type: ignore
    from torch_mlir.compiler_utils import OutputType  # type: ignore

    get_collective_info_from_torch(gm)

    print_ir = os.environ.get("MOPT_PRINT_IR") == "1"

    if print_ir:
        print("=" * 80)
        print("Stage 1: Original FX Graph")
        print("=" * 80)
        print(gm.graph)
        print()

    gm = apply_decompositions(gm, example_inputs, DEFAULT_DECOMPOSITIONS)

    if print_ir:
        print("=" * 80)
        print("Stage 2: FX Graph After Decompositions")
        print("=" * 80)
        print(gm.graph)
        print()

    stablehlo_module = fx.stateless_fx_import(gm, output_type=OutputType.STABLEHLO)

    if print_ir:
        print("=" * 80)
        print("Stage 3: StableHLO MLIR Module")
        print("=" * 80)
        print(stablehlo_module)
        print()

    stablehlo_text = mlir_module_to_text(stablehlo_module)
    mlir_module = parse_mlir_module_from_text(stablehlo_text)

    if print_ir:
        print("=" * 80)
        print("Stage 4: Re-parsed MLIR Module (before passes)")
        print("=" * 80)
        print(mlir_module)
        print()

    run_stablehlo_to_mrt_passes(mlir_module)

    if print_ir:
        print("=" * 80)
        print("Stage 5: MRT Dialect Module (after passes)")
        print("=" * 80)
        print(mlir_module)
        print()

    executor, placeholder_nodes = gen_executor_from_mlir_module(mlir_module)

    if print_ir:
        print("=" * 80)
        print("Stage 6: Executor Graph")
        print("=" * 80)
        executor.dump_graph()
        print()

    executor.build()
    return create_compiled_callable(executor, placeholder_nodes)


def create_compiled_callable(executor, placeholder_nodes):
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
