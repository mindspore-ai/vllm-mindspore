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
- An FX backend entry that applies decompositions and imports to MRT.
- Integration with MRT dialect through the _executor_builder module.
"""

import os
import io
import re
import datetime
from contextlib import redirect_stdout
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

try:
    import mfusion  # pylint: disable=import-outside-toplevel,unused-import

    MFUSION_AVAILABLE = True
except ImportError:
    MFUSION_AVAILABLE = False


def _is_print_ir_enabled() -> bool:
    return os.environ.get("MOPT_PRINT_IR") == "1"


def _sanitize_filename(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s[:180] if len(s) > 180 else s


class _DumpContext:
    """Per-compilation dump manager (stores per-stage IR/graphs to disk)."""

    def __init__(self, root_dir: str):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(os.path.abspath(root_dir), f"run_{ts}_pid{os.getpid()}")
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self._counter = 0
        self._index_path = os.path.join(run_dir, "index.txt")

    def dump_text(self, stage_title: str, text: str, *, ext: str):
        self._counter += 1
        base = f"{self._counter:02d}_{_sanitize_filename(stage_title)}"
        path = os.path.join(self.run_dir, f"{base}.{ext}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
        with open(self._index_path, "a", encoding="utf-8") as f:
            f.write(f"{self._counter:02d}\t{stage_title}\t{os.path.basename(path)}\n")


@dataclass(frozen=True)
class BackendOptions:
    """Runtime options for the FX backend.

    We read env vars once and then pass the options through the compilation flow,
    instead of scattering `os.environ.get()` checks across the pipeline logic.
    """

    print_ir: bool
    # If set, dump IR/graphs of each stage to files under this directory.
    dump_dir: Optional[str]

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
            dump_dir=env.get("MOPT_DUMP_DIR"),
        )


def _print_verbose(
    stage_title: str,
    content=None,
    dump_func=None,
    *,
    enabled: Optional[bool] = None,
    dump_ctx: Optional[_DumpContext] = None,
    dump_ext: Optional[str] = None,
):
    """Print verbose information with formatting.

    Args:
        stage_title: Title of the stage
        content: Content to print (if provided)
        dump_func: Optional callable that prints to stdout (if provided, called after printing content)
        enabled: If set, overrides MOPT_PRINT_IR.
        dump_ctx: If set, dump stage content to files.
        dump_ext: File extension hint for dumping (e.g., "mlir", "txt").
    """
    # Always dump if dump_ctx is provided (independent of print flag).
    if dump_ctx is not None:
        parts = []
        if content is not None:
            parts.append(str(content))
        if dump_func is not None:
            buf = io.StringIO()
            with redirect_stdout(buf):
                dump_func()
            parts.append(buf.getvalue())
        text = "\n".join(p for p in parts if p is not None and p != "")
        ext = dump_ext or ("mlir" if content is not None else "txt")
        dump_ctx.dump_text(stage_title, text, ext=ext)

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


def _run_pipeline(
    mlir_module,
    pass_manager,
    pipeline: str,
    *,
    stage: str,
    verbose: bool,
    dump_ctx: Optional[_DumpContext],
):
    """Run an MLIR pass pipeline and optionally dump the IR."""
    with mlir_module.context:
        pm = pass_manager.parse(pipeline)
        pm.run(mlir_module.operation)
    if dump_ctx is not None:
        dump_ctx.dump_text(f"{stage} (pipeline)", pipeline, ext="pipeline.txt")
    _print_verbose(
        stage, mlir_module, enabled=verbose, dump_ctx=dump_ctx, dump_ext="mlir"
    )


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
    from mopt import register_mrt_dialect, register_dvm_dialect

    ctx = ir.Context()
    torch_d.register_dialect(ctx)
    register_mrt_dialect(ctx)
    register_dvm_dialect(ctx)
    return ir.Module.parse(text, ctx)


def backend(
    gm: torch.fx.GraphModule, _example_inputs
):  # pylint: disable=invalid-name, unused-argument
    """FX backend entry point: FX GraphModule -> StableHLO -> MRT dialect -> GraphExecutor.

    Args:
        gm: torch.fx.GraphModule instance
        _example_inputs: (NOT USED) Example input list for type inference

    Returns:
        Callable executor function

    Environment Variables:
        MOPT_PRINT_IR: Set to 1 to print IR at each stage
        MOPT_ENABLE_LINALG_CALL: Set to 1 to lower outlined fusion regions to mrt.linalg_call (default is mrt.dvm_call)
        MOPT_DUMP_DIR: If set, dump per-stage IR/graphs to this directory.
    """
    opts = BackendOptions.from_env()
    dump_ctx = _DumpContext(opts.dump_dir) if opts.dump_dir else None
    get_collective_info_from_torch(gm)
    set_device_context()

    _print_verbose(
        "Original FX Graph",
        dump_func=gm.print_readable,
        enabled=opts.print_ir,
        dump_ctx=dump_ctx,
        dump_ext="fx.txt",
    )

    # Apply decompositions to FX GraphModule
    fake_inputs = [
        node.meta["example_value"]
        for node in gm.graph.nodes
        if node.op == "placeholder"
    ]
    m = apply_decompositions(gm, fake_inputs)
    _print_verbose(
        "FX Graph After Decompositions",
        dump_func=m.print_readable,
        enabled=opts.print_ir,
        dump_ctx=dump_ctx,
        dump_ext="fx.txt",
    )

    # Convert FX GraphModule to torch dialect MLIR module.
    # We re-parse the MLIR text here to let mopt use its own MLIR context,
    # separate from torch_mlir's. Both torch_mlir and mopt build their own
    # copy of the torch dialect, and their MLIR contexts are not shared.
    # The torch_mlir Python API uses the torch_mlir context by default,
    # which can lead to resource conflicts. By reparsing, we ensure all
    # later operations and resources are managed in mopt's context.
    mlir_module = _convert_to_torch_mlir(m)
    mlir_module = _parse_mlir_module_from_text(str(mlir_module))
    _print_verbose(
        "Torch-MLIR Raw Module (Re-parsed)",
        mlir_module,
        enabled=opts.print_ir,
        dump_ctx=dump_ctx,
        dump_ext="mlir",
    )

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
            dump_ctx=dump_ctx,
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

    # ===== MFusion Optimization =====
    if MFUSION_AVAILABLE:
        from mfusion.torch.inductor import fuse_and_optimize

        optimized_mlir_str = fuse_and_optimize(str(mlir_module))
        mlir_module = _parse_mlir_module_from_text(optimized_mlir_str)
        _print_verbose(
            "MFusion Processed Module",
            mlir_module,
            enabled=opts.print_ir,
            dump_ctx=dump_ctx,
            dump_ext="mlir",
        )

    # ===== Final Conversion =====
    # Convert remaining Torch backend ops to MRT dialect and reconcile type casts.
    run_pipeline(
        "builtin.module(convert-torch-to-mrt,reconcile-unrealized-casts,canonicalize)",
        stage="MRT Dialect Module",
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
    _print_verbose(
        "Executor Graph",
        dump_func=executor.dump_graph,
        enabled=opts.print_ir,
        dump_ctx=dump_ctx,
        dump_ext="executor.txt",
    )

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
                device_str = (
                    "npu" if device_type_str in ("npu", "privateuse1") else "cpu"
                )
                device_index = getattr(device, "index", -1)
                if device_index is None:
                    device_index = -1
                return device_str, device_index

    # default to cpu
    return "cpu", -1
