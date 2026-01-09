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
from collections import namedtuple
from dataclasses import dataclass
from typing import Mapping, Optional

import torch

from mrt.torch.utils import (
    get_collective_info_from_torch,
    set_device_context,
    to_torch,
    update_runtime_inputs,
)
from mrt.torch._decompositions import apply_decompositions
from mrt.torch._executor_builder import ExecutorBuilder


def _is_print_ir_enabled() -> bool:
    return os.environ.get("MOPT_PRINT_IR") == "1"


@dataclass(frozen=True)
class BackendOptions:
    """Runtime options for the FX backend.

    We read env vars once and then pass the options through the compilation flow,
    instead of scattering `os.environ.get()` checks across the pipeline logic.
    """

    print_ir: bool
    # Default fusion lowering is mrt.dvm_call. Set env var to opt into linalg_call.
    enable_linalg_call: bool

    @staticmethod
    def _flag(env: Mapping[str, str], name: str, default: bool = False) -> bool:
        # Convention: "1" enables; missing/other values disable.
        val = env.get(name)
        if val is None:
            return default
        return val == "1"

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "BackendOptions":
        if env is None:
            env = os.environ
        return cls(
            print_ir=cls._flag(env, "MOPT_PRINT_IR", default=False),
            enable_linalg_call=cls._flag(env, "MOPT_ENABLE_LINALG_CALL", default=False),
        )


def _print_verbose(stage_title: str, content=None, dump_func=None, *, enabled: Optional[bool] = None):
    """Print verbose information with formatting.

    Args:
        stage_title: Title of the stage
        content: Content to print (if provided)
        dump_func: Optional callable that prints to stdout (if provided, called after printing content)
        enabled: If set, overrides MOPT_PRINT_IR.
    """
    if enabled is None:
        enabled = _is_print_ir_enabled()
    if not enabled:
        return
    print("=" * 80)
    print(stage_title)
    print("=" * 80)
    if content is not None:
        print(content)
    if dump_func is not None:
        dump_func()
    print()


def _run_pipeline(mlir_module, pass_manager, pipeline: str, *, stage: str, verbose: bool):
    """Run an MLIR pass pipeline and optionally dump the IR."""
    with mlir_module.context:
        pm = pass_manager.parse(pipeline)
        pm.run(mlir_module.operation)
    _print_verbose(stage, mlir_module, enabled=verbose)


def _pipeline_outline_stablehlo_fusion_regions(*, enable_dvm_call: bool) -> str:
    outline_opts = ""
    if enable_dvm_call:
        # DVM-call mode: allow dot_general (matmul) and dynamic shapes to be outlined.
        outline_opts = "{allow-dot-general=true allow-dynamic-shape=true}"
    return f"builtin.module(outline-stablehlo-fusion-regions{outline_opts})"


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


def _has_fusible_ops(mlir_module) -> bool:
    """Check if any op in the module is tagged for StableHLO conversion (fusible ops)."""
    module_op = mlir_module.operation
    for region in module_op.regions:
        for block in region:
            for op in block:
                if _check_op_tag_recursive(op):
                    return True
    return False


def _check_op_tag_recursive(op) -> bool:
    if "mopt.torch_to_stablehlo" in op.attributes:
        return True
    for region in op.regions:
        for block in region:
            for child_op in block:
                if _check_op_tag_recursive(child_op):
                    return True
    return False


def backend(gm: torch.fx.GraphModule, _example_inputs):  # pylint: disable=invalid-name, unused-argument
    """FX backend entry point: FX GraphModule -> StableHLO -> MRT dialect -> GraphExecutor.

    Args:
        gm: torch.fx.GraphModule instance
        _example_inputs: (NOT USED) Example input list for type inference

    Returns:
        Callable executor function

    Environment Variables:
        MOPT_PRINT_IR: Set to 1 to print IR at each stage
        MOPT_ENABLE_LINALG_CALL: Set to 1 to lower outlined fusion regions to mrt.linalg_call (default is mrt.dvm_call)
    """
    opts = BackendOptions.from_env()
    get_collective_info_from_torch(gm)
    set_device_context()

    _print_verbose("Original FX Graph", dump_func=gm.print_readable, enabled=opts.print_ir)

    # Apply decompositions to FX GraphModule
    fake_inputs = [
        node.meta["example_value"]
        for node in gm.graph.nodes
        if node.op == "placeholder"
    ]
    m = apply_decompositions(gm, fake_inputs)
    _print_verbose("FX Graph After Decompositions", dump_func=m.print_readable, enabled=opts.print_ir)

    # Convert FX GraphModule to torch dialect MLIR module.
    # We re-parse the MLIR text here to let mopt use its own MLIR context,
    # separate from torch_mlir's. Both torch_mlir and mopt build their own
    # copy of the torch dialect, and their MLIR contexts are not shared.
    # The torch_mlir Python API uses the torch_mlir context by default,
    # which can lead to resource conflicts. By reparsing, we ensure all
    # later operations and resources are managed in mopt's context.
    mlir_module = _convert_to_torch_mlir(m)
    mlir_module = _parse_mlir_module_from_text(str(mlir_module))
    _print_verbose("Torch-MLIR Raw Module (Re-parsed)", mlir_module, enabled=opts.print_ir)

    # Run pass pipeline to convert torch_mlir RAW to TORCH backend
    # pylint: disable=import-outside-toplevel
    # Reason: mopt must be imported here to prevent default loading at module level
    from mopt.passmanager import PassManager

    def run_pipeline(pipeline: str, *, stage: str):
        _run_pipeline(
            mlir_module,
            PassManager,
            pipeline,
            stage=stage,
            verbose=opts.print_ir,
        )

    # ===== Conversion Pipeline =====
    # Torch RAW -> Torch Backend IR
    run_pipeline(
        "builtin.module("
        + "torchdynamo-export-to-torch-backend-pipeline{decompose-complex-ops=false}"
        + ",func.func(torch-decompose-complex-ops,canonicalize)"
        + ")",
        stage="Torch Backend IR",
    )

    # Mark Torch ops to be converted to StableHLO based on whitelist/rules.
    # This pass identifies and marks operations that can be fused.
    run_pipeline(
        "builtin.module(mark-torch-to-stablehlo-op)",
        stage="After Marking Torch To StableHLO Ops",
    )

    # Check if any op is marked. If not, skip StableHLO conversion entirely.
    if _has_fusible_ops(mlir_module):
        # Torch Backend -> StableHLO
        run_pipeline(
            "builtin.module(torch-backend-to-stablehlo-backend-pipeline)",
            stage="StableHLO IR (from whitelisted ops)",
        )

        # Fusion pipeline:
        # Outline fusible regions (DVM-call mode enables advanced outlining).
        use_dvm_outline = not opts.enable_linalg_call
        run_pipeline(
            _pipeline_outline_stablehlo_fusion_regions(enable_dvm_call=use_dvm_outline),
            stage="StableHLO After Outlining Fusion Regions",
        )

        if opts.enable_linalg_call:
            # Prepare outlined fusion calls:
            # - Serialize outlined fusion functions to Linalg MLIR text (with hacc.entry / hacc.function_kind<HOST>)
            # - Annotate corresponding func.call ops with the serialized text
            # - convert to mrt.linalg_call.
            run_pipeline(
                "builtin.module(convert-outlined-fusion-call,symbol-dce)",
                stage="After Annotating Outlined Fusion Calls",
            )
        else:
            # Alternative lowering for outlined StableHLO clusters:
            # StableHLO (outlined) -> DVM -> mrt.dvm_call (payload JSON)
            run_pipeline(
                "builtin.module(convert-stablehlo-to-dvm,convert-dvm-to-mrt-dvm-call,symbol-dce)",
                stage="After Converting Outlined Regions to mrt.dvm_call",
            )

        # Convert remaining StableHLO to MRT dialect
        run_pipeline(
            "builtin.module(convert-stablehlo-to-mrt)",
            stage="MRT Dialect Module (after convert-stablehlo-to-mrt)",
        )

    # Convert remaining Torch backend ops to MRT dialect.
    # This is always needed: in fusion mode, StableHLO conversion only covers a subset of ops,
    # so the module can still contain Torch backend dialect operations.
    run_pipeline(
        "builtin.module(convert-torch-to-mrt,reconcile-unrealized-casts,canonicalize)",
        stage="MRT Dialect Module (after convert-torch-to-mrt)",
    )

    # Run pass with device information via PassOptions
    device_str, device_index = _get_device_info_from_gm(gm)
    run_pipeline(
        f"builtin.module(set-tensor-device{{device-type={device_str} device-index={device_index}}},"
        "reconcile-unrealized-casts)",
        stage="MRT Dialect Module (after set-tensor-device pass)",
    )

    # Build MRT dialect module into GraphExecutor
    executor, param_nodes = ExecutorBuilder().build(mlir_module, fake_inputs)
    _print_verbose("Executor Graph", dump_func=executor.dump_graph, enabled=opts.print_ir)

    # Create compiled callable from GraphExecutor
    executor.build()

    def compiled_callable(*inputs: torch.Tensor):
        set_device_context()
        update_runtime_inputs(param_nodes, inputs)
        result = executor.run()
        return to_torch(result)

    return compiled_callable


def _get_device_info_from_gm(gm: torch.fx.GraphModule):
    """Extract device information from input node metadata of FX GraphModule."""
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            # 'val' is the standard metadata (usually FakeTensor) produced by TorchDynamo/AOTAutograd
            # 'example_value' is the metadata produced by conventional FX tracing
            val = node.meta.get("val", node.meta.get("example_value"))
            if isinstance(val, torch.Tensor):
                device = val.device
                device_type_str = str(device.type)
                # Handle device type string
                device_str = "npu" if device_type_str in ("npu", "privateuse1") else "cpu"
                device_index = getattr(device, "index", -1)
                if device_index is None:
                    device_index = -1
                return device_str, device_index

    # default to cpu
    return "cpu", -1
