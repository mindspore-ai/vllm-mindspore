# Copyright 2026 Huawei Technologies Co., Ltd
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

"""DVM v2 adapter helpers for the torch.fx backend."""

import ctypes
import importlib
import os
from pathlib import Path
from typing import List, Optional, Tuple

from torch.fx.node import Argument, Node

from ms_inferrt.ir import Op


# Registry for DVM V2 callable kernels. InferRT IR carries only the string
# handle; the Python registry owns the materialized kobj so C++ operators do
# not hold Python objects until interpreter teardown.
_DVM_FUNC_REGISTRY = {}
_DVM_KERNEL_REGISTRY = {}
_DVM_FUNC_COUNTER = 0
_DVM_COMPILED_KERNEL_NAME_PREFIX = "dvm_"
_DVM_V2_LIBRARY_NAME = "libops_ascend_dvm_v2.so"
_DVM_V2_LOADED = False
_DVM_V2_LOAD_ERROR = None
_DVM_V2_CDLL = None
_DVM_TEMPLATE_REQUIRED_ATTRS = ("is_split", "kernel_type", "kernel_flags")
_DVM_KERNEL_REQUIRED_ATTRS = (
    "kernel",
    "relocs",
    "loads",
    "stores",
    "num_tensor_inputs",
    "num_outputs",
    "workspace_size",
    "is_dynamic",
    "is_split",
)


def _dvm_v2_library_path() -> Path:
    import ms_inferrt  # pylint: disable=import-outside-toplevel

    return Path(ms_inferrt.__file__).resolve().parent / "lib" / _DVM_V2_LIBRARY_NAME


def ensure_dvm_v2_runtime_available():
    """Load the optional DVM v2 op library only when a DVM graph needs it."""
    global _DVM_V2_LOADED, _DVM_V2_LOAD_ERROR, _DVM_V2_CDLL  # pylint: disable=global-statement
    if _DVM_V2_LOADED:
        return

    library_path = _dvm_v2_library_path()
    if not library_path.exists():
        _DVM_V2_LOAD_ERROR = (
            f"DVM v2 support is not built in this ms_inferrt package: "
            f"{library_path} does not exist"
        )
        raise RuntimeError(_DVM_V2_LOAD_ERROR)

    mode = getattr(os, "RTLD_LAZY", 1) | getattr(os, "RTLD_LOCAL", 0)
    try:
        _DVM_V2_CDLL = ctypes.CDLL(str(library_path), mode=mode)
    except OSError as exc:
        _DVM_V2_LOAD_ERROR = f"Failed to load DVM v2 op library {library_path}: {exc}"
        raise RuntimeError(_DVM_V2_LOAD_ERROR) from exc

    _DVM_V2_LOADED = True
    _DVM_V2_LOAD_ERROR = None


def _require_attrs(obj, attrs, what: str):
    missing = [attr for attr in attrs if not hasattr(obj, attr)]
    if missing:
        raise RuntimeError(
            f"Current torch-npu DVM kernel object does not provide InferRT DVM v2 ABI for {what}; "
            f"missing attributes: {', '.join(missing)}"
        )


def _materialize_dvm_kernel(dvm_func):
    """Instantiate and setup an isolated DVM kernel object for one InferRT op."""
    template = getattr(dvm_func, "kobj", None)
    builder = getattr(dvm_func, "__wrapped__", None)
    if template is None or builder is None:
        raise RuntimeError(
            "DVM function must be a torch_npu dvm.kernel decorated function "
            "with both 'kobj' and '__wrapped__' attributes"
        )
    _require_attrs(template, _DVM_TEMPLATE_REQUIRED_ATTRS, "kernel template")

    kernel_cls = type(template)
    if template.is_split():
        kobj = kernel_cls()
    else:
        kobj = kernel_cls(template.kernel_type(), template.kernel_flags())
    builder(kobj)
    kobj.setup()
    if kobj is None:
        raise RuntimeError("DVM function did not produce a kernel object")
    _require_attrs(kobj, _DVM_KERNEL_REQUIRED_ATTRS, "materialized kernel")
    return kobj


def register_dvm_func(dvm_func) -> str:
    """Register a decorated dvm.kernel function and return its handle."""
    global _DVM_FUNC_COUNTER  # pylint: disable=global-statement
    ensure_dvm_v2_runtime_available()
    handle = f"dvm_func_{_DVM_FUNC_COUNTER}"
    _DVM_FUNC_COUNTER += 1
    _DVM_FUNC_REGISTRY[handle] = dvm_func
    _DVM_KERNEL_REGISTRY[handle] = _materialize_dvm_kernel(dvm_func)
    return handle


def get_dvm_kernel_obj(handle: str):
    """Get a materialized dvm.kernel object by handle."""
    return _DVM_KERNEL_REGISTRY.get(handle)


def is_compiled_kernel_wrapper_mutation_target(target) -> bool:
    """Return whether an FX node target is the compiled-kernel mutation HOP."""
    target_name = getattr(target, "__name__", None)
    if target_name == "compiled_kernel_wrapper_mutation":
        return True
    return str(target) == "torch.ops.higher_order.compiled_kernel_wrapper_mutation"


def _get_compiled_kernel_name(compiled_kernel) -> Optional[str]:
    """Resolve the generated kernel name from a compiled-kernel wrapper."""
    if compiled_kernel is None:
        return None
    for candidate in (compiled_kernel, getattr(compiled_kernel, "_compiled", None)):
        if candidate is None:
            continue
        kernel_name = getattr(candidate, "_kernel_name", None)
        if isinstance(kernel_name, str) and kernel_name:
            return kernel_name
        kernel_name = getattr(candidate, "__name__", None)
        if isinstance(kernel_name, str) and kernel_name:
            return kernel_name
    return None


def _resolve_compiled_kernel_from_side_table(kernel_idx: int):
    """Look up a compiled kernel from torch_dispatch_capture's side table."""
    try:
        module = importlib.import_module("torch_dispatch_capture.v4.compiled_kernel_hop")
    except ImportError as exc:
        raise RuntimeError(
            "compiled_kernel_wrapper_mutation requires torch_dispatch_capture.v4.compiled_kernel_hop"
        ) from exc
    return module.compiled_kernel_side_table.get_kernel(kernel_idx)


def _extract_dvm_func_from_compiled_kernel(compiled_kernel):
    """Return the decorated DVM function from a compiled-kernel wrapper."""
    kernel_name = _get_compiled_kernel_name(compiled_kernel)
    if kernel_name is None or not kernel_name.startswith(_DVM_COMPILED_KERNEL_NAME_PREFIX):
        return None

    candidates = [compiled_kernel, getattr(compiled_kernel, "_compiled", None)]
    for candidate in candidates:
        if candidate is None or not callable(candidate):
            continue
        if hasattr(candidate, "kobj"):
            return candidate
    for candidate in candidates:
        if candidate is None or not callable(candidate):
            continue
        return candidate
    return None


def get_dvm_func_from_node(node: Node):
    """Resolve the decorated DVM function represented by an FX node."""
    if not is_compiled_kernel_wrapper_mutation_target(node.target):
        return None

    kernel_idx = node.kwargs.get("kernel_idx", None)
    if not isinstance(kernel_idx, int):
        raise RuntimeError(
            "compiled_kernel_wrapper_mutation requires integer kwargs['kernel_idx']"
        )

    compiled_kernel = _resolve_compiled_kernel_from_side_table(kernel_idx)
    dvm_func = _extract_dvm_func_from_compiled_kernel(compiled_kernel)
    if dvm_func is None:
        return None

    return dvm_func


def get_compiled_kernel_mutation_args(node: Node) -> Tuple[Tuple[Argument, ...], Tuple[int, ...]]:
    """Extract explicit args and mutated output indices from a mutation HOP node."""
    if not is_compiled_kernel_wrapper_mutation_target(node.target):
        raise RuntimeError("Expected compiled_kernel_wrapper_mutation node")

    hop_args = node.kwargs.get("args", None)
    mutated_arg_indices = node.kwargs.get("mutated_arg_indices", ())
    if not isinstance(hop_args, tuple):
        raise RuntimeError("compiled_kernel_wrapper_mutation requires tuple kwargs['args']")
    if not isinstance(mutated_arg_indices, tuple) or not all(
        isinstance(index, int) for index in mutated_arg_indices
    ):
        raise RuntimeError(
            "compiled_kernel_wrapper_mutation requires tuple[int, ...] kwargs['mutated_arg_indices']"
        )
    return hop_args, mutated_arg_indices


def _get_compiled_kernel_mutation_output_nodes(node: Node) -> List[Node]:
    """Return FX nodes that represent mutated output buffers."""
    hop_args, mutated_arg_indices = get_compiled_kernel_mutation_args(node)
    output_nodes = []
    for index in mutated_arg_indices:
        if index < 0 or index >= len(hop_args):
            raise RuntimeError(
                f"compiled_kernel_wrapper_mutation mutated arg index {index} is out of range for {len(hop_args)} args"
            )
        output_node = hop_args[index]
        if not isinstance(output_node, Node):
            raise RuntimeError(
                f"compiled_kernel_wrapper_mutation mutated arg[{index}] must be an FX node, got {type(output_node)}"
            )
        output_nodes.append(output_node)
    return output_nodes


def _map_args(args, env, executor, sym_mgr) -> List[Node]:
    """Map DVM HOP arguments to InferRT graph nodes."""

    def _map_arg(arg):
        if isinstance(arg, Node):
            return env[arg]
        if isinstance(arg, (list, tuple)):
            return executor.make_tuple([_map_arg(item) for item in arg])
        return executor.add_value_node(sym_mgr.from_torch_with_sym(arg))

    return [_map_arg(arg) for arg in args]


def _prepare_dvm_call_v2_args(node: Node, dvm_func, executor, env, sym_mgr) -> List[Node]:
    """Build InferRT dvm_call_v2 inputs from a compiled-kernel mutation HOP node."""
    hop_args, mutated_arg_indices = get_compiled_kernel_mutation_args(node)
    mutated_arg_index_set = set(mutated_arg_indices)
    flat_node_args = [
        arg for idx, arg in enumerate(hop_args) if idx not in mutated_arg_index_set
    ]
    handle = register_dvm_func(dvm_func)
    return _map_args([handle] + flat_node_args, env, executor, sym_mgr)


def lower_compiled_kernel_dvm_node(
    node,
    executor,
    env,
    sym_mgr,
    get_node_meta_value,
    add_tuple_getitem_node,
):
    """Lower a DVM compiled-kernel mutation HOP node into an InferRT dvm_call_v2 op."""
    dvm_func = get_dvm_func_from_node(node)
    if dvm_func is None:
        return False

    output_nodes = _get_compiled_kernel_mutation_output_nodes(node)
    if not output_nodes:
        raise RuntimeError("DVM compiled_kernel_wrapper_mutation requires at least one mutated output buffer")

    input_nodes = _prepare_dvm_call_v2_args(node, dvm_func, executor, env, sym_mgr)

    output_examples = []
    for output_node in output_nodes:
        example_value = get_node_meta_value(output_node)
        if example_value is None:
            raise RuntimeError(
                f"DVM compiled kernel output buffer node '{output_node.name}' is missing example_value/val metadata"
            )
        output_examples.append(example_value)

    if len(output_examples) == 1:
        output_value = sym_mgr.from_torch_with_sym(output_examples[0])
        dvm_node = executor.add_op_node(Op.dvm_call_v2, input_nodes, output_value)
        env[node] = dvm_node
        env[output_nodes[0]] = dvm_node
        return True

    tuple_output = sym_mgr.from_torch_with_sym(tuple(output_examples))
    tuple_node = executor.add_op_node(Op.dvm_call_v2, input_nodes, tuple_output)
    env[node] = tuple_node
    for idx, (output_node, output_example) in enumerate(zip(output_nodes, output_examples)):
        output_value = sym_mgr.from_torch_with_sym(output_example)
        env[output_node] = add_tuple_getitem_node(executor, sym_mgr, tuple_node, idx, output_value)
    return True
