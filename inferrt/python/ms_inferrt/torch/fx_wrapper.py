# Copyright(c) 2026, the respective contributors
# All rights reserved.
#
# Modifications by Huawei Technologies Co., Ltd. 2025.
#
# This file contains code derived from PyTorch:
# torch/_higher_order_ops/triton_kernel_wrap.py
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC
#    Laboratories America and IDIAP Research Institute nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Inductor fx_wrapper integration for ms_inferrt.

Importing this module installs an fx_wrapper codegen for torch-npu when that
backend is available. Users still enable the official inductor path with
``torch._inductor.config.patch({"fx_wrapper": True})``; the installed wrapper
then routes the generated host FX GraphModule to ms_inferrt by default.

The fused-kernel HOP pieces are migrated from the temporary TDC v4 prototype so
torch-npu/dvm fused kernels can survive inside the host FX graph instead of
being rejected by stock ``WrapperFxCodegen``.
"""
# pylint: disable=import-outside-toplevel,unused-argument,protected-access,undefined-all-variable,wrong-import-position

from __future__ import annotations

import abc
import contextlib
import dataclasses
import os
from collections.abc import Callable
from typing import Any

import torch
from torch._inductor.codecache import LambdaFuture, PyCodeCache
from torch._inductor.codegen.common import FileBackedGraphModule
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.codegen.wrapper_fxir import FxConverter, WrapperFxCodegen
from torch._inductor.runtime.triton_heuristics import CachingAutotuner

from ms_inferrt.torch import compiled_kernel_hop as _compiled_kernel_hop

CompiledKernelSideTable = _compiled_kernel_hop.CompiledKernelSideTable
compiled_kernel_side_table = _compiled_kernel_hop.compiled_kernel_side_table
compiled_kernel_wrapper_functional = (
    _compiled_kernel_hop.compiled_kernel_wrapper_functional
)
compiled_kernel_wrapper_mutation = _compiled_kernel_hop.compiled_kernel_wrapper_mutation
launch_compiled_kernel_functional = (
    _compiled_kernel_hop.launch_compiled_kernel_functional
)
launch_compiled_kernel_mutation = _compiled_kernel_hop.launch_compiled_kernel_mutation


_active_fx_backend: "Callable | None" = None
_installed_devices: set[str] = set()


def _default_fx_backend(gm, example_inputs):
    """Run the default ms_inferrt FX backend."""

    from ms_inferrt.torch.fx_backend import backend

    return backend(gm, example_inputs)


def get_fx_wrapper_backend() -> "Callable | None":
    """Return the configured process-wide fx_wrapper backend.

    ``None`` means the wrapper will use ``ms_inferrt.torch.fx_backend.backend``.
    """

    return _active_fx_backend


def set_fx_wrapper_backend(gm_backend: "Callable | None") -> "Callable | None":
    """Set the process-wide backend used by the installed fx_wrapper.

    Args:
        gm_backend: callable ``(gm, example_inputs) -> compiled_callable``.
            Passing ``None`` restores the default ms_inferrt backend.

    Returns:
        The previous backend value.
    """

    global _active_fx_backend
    previous = _active_fx_backend
    _active_fx_backend = gm_backend
    return previous


@contextlib.contextmanager
def fx_wrapper_backend(gm_backend: "Callable | None"):
    """Temporarily set the fx_wrapper backend."""

    previous = set_fx_wrapper_backend(gm_backend)
    try:
        yield
    finally:
        set_fx_wrapper_backend(previous)


def _resolve_backend() -> Callable:
    """Return the configured backend or the default ms_inferrt backend."""

    return _active_fx_backend or _default_fx_backend


class CompiledKernelBackend(abc.ABC):
    """Teaches the FX converter how to preserve one non-Triton kernel kind."""

    @abc.abstractmethod
    def handles_definition(self, defn_line) -> bool:
        """Return true if this backend owns the kernel definition line."""

    @abc.abstractmethod
    def compile_kernel(self, converter: "CompiledKernelFxConverter", defn_line) -> Callable:
        """Compile to a callable taking flat positional args."""

    @abc.abstractmethod
    def mutated_arg_indices(self, call_line) -> tuple[int, ...]:
        """Return positions in the kernel call args written by the kernel."""


_COMPILED_BACKENDS: list[CompiledKernelBackend] = []


def register_compiled_kernel_backend(backend: CompiledKernelBackend) -> None:
    """Register a compiled-kernel backend."""

    if not any(isinstance(x, type(backend)) for x in _COMPILED_BACKENDS):
        _COMPILED_BACKENDS.append(backend)


def _select_backend(defn_line) -> "CompiledKernelBackend | None":
    """Return the registered backend that owns a compiled-kernel definition."""

    for backend in _COMPILED_BACKENDS:
        if backend.handles_definition(defn_line):
            return backend
    return None


class CompiledKernelFxConverter(FxConverter):
    """FxConverter that routes non-Triton kernels through registered backends."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.compiled_kernels: dict[str, tuple[int, CompiledKernelBackend]] = {}

    def _generate_kernel_definition(self, line) -> None:
        backend = _select_backend(line)
        if backend is None:
            super()._generate_kernel_definition(line)
            return
        kernel = backend.compile_kernel(self, line)
        idx = compiled_kernel_side_table.add_kernel(kernel)
        self.compiled_kernels[line.kernel_name] = (idx, backend)

    def _generate_kernel_call(self, line) -> None:
        entry = self.compiled_kernels.get(line.kernel_name)
        if entry is None:
            super()._generate_kernel_call(line)
            return
        idx, backend = entry
        self.gm.graph.call_function(
            compiled_kernel_wrapper_mutation,
            kwargs={
                "kernel_idx": idx,
                "mutated_arg_indices": tuple(backend.mutated_arg_indices(line)),
                "args": tuple(self._lookup_args(line.call_args)),
            },
        )


class InferrtFxWrapper(WrapperFxCodegen):
    """WrapperFxCodegen that sends the host FX graph to the configured backend."""

    def _generate(self, is_inference: bool):
        """Generate the wrapper FX graph and compile it with the active backend."""

        self.run_wrapper_ir_passes(is_inference)
        prologue = "\n".join([self.imports.getvalue(), self.header.getvalue()])
        gm = CompiledKernelFxConverter(
            lines=self.lines,
            prologue=prologue,
            graph_inputs=self.get_fx_graph_inputs(),
            graph_outputs=self.get_graph_outputs(),
            subgms=self.subgms,
            is_subgraph=self.is_subgraph,
        ).generate()
        return FileBackedGraphModule(gm, self.compile_graph(gm)), None

    def compile_graph(self, gm):
        """Compile the generated FX graph with the active ms_inferrt backend."""

        for node in gm.graph.nodes:
            if "example_value" not in node.meta and node.meta.get("val") is not None:
                node.meta["example_value"] = node.meta["val"]
        example_inputs = [
            n.meta["val"] for n in gm.graph.nodes if n.op == "placeholder"
        ]
        return _resolve_backend()(gm, example_inputs)


def cpp_mutated_arg_indices(arg_types) -> tuple[int, ...]:
    """A C ABI kernel arg is mutated iff it is a non-const pointer."""

    return tuple(
        i
        for i, t in enumerate(arg_types)
        if isinstance(t, str) and "*" in t and not t.strip().startswith("const")
    )


class CppPybindingBackend(CompiledKernelBackend):
    """Inductor CPU cpp_pybinding kernels."""

    def handles_definition(self, defn_line) -> bool:
        return not getattr(defn_line, "gpu", True)

    def compile_kernel(self, converter, defn_line) -> Callable:
        code = PythonWrapperCodegen._format_kernel_definition(
            defn_line.kernel_name, defn_line.kernel_body, metadata=defn_line.metadata
        )
        mod = PyCodeCache.load("\n".join([converter.prologue, code]))
        kernel = getattr(mod, defn_line.kernel_name)
        if isinstance(kernel, LambdaFuture):
            kernel = kernel.result()
        if isinstance(kernel, CachingAutotuner):
            raise AssertionError("Triton kernel reached the compiled-kernel backend")
        return kernel

    def mutated_arg_indices(self, call_line) -> tuple[int, ...]:
        return cpp_mutated_arg_indices(call_line.arg_types)


from ms_inferrt.torch.dvm_fx_wrapper import DvmBackend

register_compiled_kernel_backend(CppPybindingBackend())
register_compiled_kernel_backend(DvmBackend())


def _replace_fx_wrapper_codegen(device_codegen, wrapper_cls):
    """Return a device codegen with the requested fx_wrapper class installed."""

    try:
        return dataclasses.replace(device_codegen, fx_wrapper_codegen=wrapper_cls)
    except (TypeError, AttributeError):
        if not hasattr(device_codegen, "fx_wrapper_codegen"):
            raise
        device_codegen.fx_wrapper_codegen = wrapper_cls
        return device_codegen


def install_torch_npu_fx_wrapper(
    gm_backend: "Callable | None" = None, device: str = "npu"
) -> bool:
    """Install the ms_inferrt fx_wrapper codegen for torch-npu.

    This is idempotent. It imports ``torch_npu._inductor`` when available so the
    torch-npu backend has registered its device codegen before we patch it.
    Returns ``True`` when the wrapper is installed, otherwise ``False`` when
    torch-npu/inductor is not available in the current environment.
    """

    if gm_backend is not None:
        set_fx_wrapper_backend(gm_backend)

    try:
        import torch_npu  # pylint: disable=unused-import,import-outside-toplevel
        import torch_npu._inductor  # pylint: disable=unused-import,import-outside-toplevel
        from torch._inductor.codegen.common import (
            device_codegens,
            init_backend_registration,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if os.environ.get("MS_INFERRT_FX_WRAPPER_DEBUG") == "1":
            print(f"ms_inferrt fx_wrapper install skipped: {exc}")
        return False

    device = torch.device(device).type
    init_backend_registration()
    device_codegen = device_codegens.get(device)
    if device_codegen is None:
        if os.environ.get("MS_INFERRT_FX_WRAPPER_DEBUG") == "1":
            print(f"ms_inferrt fx_wrapper install skipped: no backend for {device}")
        return False

    if getattr(device_codegen, "fx_wrapper_codegen", None) is InferrtFxWrapper:
        _installed_devices.add(device)
        return True

    device_codegens[device] = _replace_fx_wrapper_codegen(
        device_codegen, InferrtFxWrapper
    )
    _installed_devices.add(device)
    return True


@contextlib.contextmanager
def enable_torch_npu_fx_wrapper(
    gm_backend: "Callable | None" = None,
    device: str = "npu",
    *,
    patch_config: bool = True,
):
    """Convenience context for tests and probes.

    Production callers may use the official inductor API directly:
    ``torch._inductor.config.patch({"fx_wrapper": True})``.
    """

    install_torch_npu_fx_wrapper(gm_backend=gm_backend, device=device)
    if not patch_config:
        yield
        return

    import torch._inductor.config as inductor_config

    with inductor_config.patch(
        {"fx_wrapper": True, "size_asserts": False, "alignment_asserts": False}
    ):
        yield


_EXPORTED_NAMES = (
    "CompiledKernelBackend",
    "CompiledKernelFxConverter",
    "CompiledKernelSideTable",
    "CppPybindingBackend",
    "DvmBackend",
    "InferrtFxWrapper",
    "launch_compiled_kernel_functional",
    "launch_compiled_kernel_mutation",
    "compiled_kernel_side_table",
    "compiled_kernel_wrapper_functional",
    "compiled_kernel_wrapper_mutation",
    "enable_torch_npu_fx_wrapper",
    "fx_wrapper_backend",
    "get_fx_wrapper_backend",
    "install_torch_npu_fx_wrapper",
    "register_compiled_kernel_backend",
    "set_fx_wrapper_backend",
)

__all__ = [name for name in _EXPORTED_NAMES if name in globals()]


install_torch_npu_fx_wrapper()
