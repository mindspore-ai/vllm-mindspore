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

"""
FX graph pass: eliminate no-op dtype casts.

InferRT lowers torch dtype conversions to Op.cast, which calls aclnnCast on
Ascend. When the cast output dtype is identical to the input dtype, the cast has
no numeric effect and can be replaced by its input tensor.
"""

import os
from typing import Any

import torch
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node


_DISABLE_CAST_ELIMINATION_ENV = "MS_INFERRT_DISABLE_CAST_ELIMINATION"

_DTYPE_METHOD_TARGETS = {
    "bfloat16": torch.bfloat16,
    "bool": torch.bool,
    "byte": torch.uint8,
    "char": torch.int8,
    "double": torch.float64,
    "float": torch.float32,
    "half": torch.float16,
    "int": torch.int32,
    "long": torch.int64,
    "short": torch.int16,
}

_ATEN_TO_TARGETS = {
    torch.ops.aten.to.device,
    torch.ops.aten.to.dtype,
    torch.ops.aten.to.other,
    torch.ops.aten.to.dtype_layout,
    torch.ops.aten.to.prim_Device,
    torch.ops.aten.to.prim_dtype,
    torch.ops.aten.to.prim_other,
}

# aten._to_copy is intentionally excluded because it must return a non-aliasing
# copy even when the dtype and device are unchanged.
_CAST_FUNCTION_TARGETS = {
    torch.ops.prims.convert_element_type.default,
    *_ATEN_TO_TARGETS,
}

_SCALAR_TYPE_TO_DTYPE = {
    0: torch.uint8,
    1: torch.int8,
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64,
    11: torch.bool,
    15: torch.bfloat16,
}


def _example_value(value: Any) -> Any:
    if isinstance(value, Node):
        return value.meta.get("example_value", value.meta.get("val", None))
    return value


def _dtype_of(value: Any) -> torch.dtype | None:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, int):
        return _SCALAR_TYPE_TO_DTYPE.get(value)
    example_value = _example_value(value)
    return getattr(example_value, "dtype", None)


def _as_device(value: Any) -> torch.device | None:
    """Coerce a value into a torch.device, returning None on failure."""
    if value is None:
        return None
    if isinstance(value, torch.device):
        return value
    if isinstance(value, str):
        try:
            return torch.device(value)
        except (TypeError, RuntimeError):
            return None
    return None


def _device_of(value: Any) -> torch.device | None:
    device = _as_device(value)
    if device is not None:
        return device
    example_value = _example_value(value)
    return getattr(example_value, "device", None)


def _layout_of(value: Any) -> torch.layout | None:
    example_value = _example_value(value)
    return getattr(example_value, "layout", None)


def _is_tensor_like_node(value: Any) -> bool:
    example_value = _example_value(value)
    return isinstance(value, Node) and hasattr(example_value, "dtype")


def _kw_or_arg(node: Node, key: str, arg_index: int, default: Any = None) -> Any:
    if key in node.kwargs:
        return node.kwargs[key]
    if len(node.args) > arg_index:
        return node.args[arg_index]
    return default


def _has_explicit_copy(node: Node) -> bool:
    """Return True when the node explicitly requests a copy (copy=True)."""
    if "copy" in node.kwargs:
        return bool(node.kwargs["copy"])

    # Map each aten.to overload to the positional index of its copy argument.
    copy_arg_index = {
        torch.ops.aten.to.dtype: 3,
        torch.ops.aten.to.other: 3,
        torch.ops.aten.to.prim_dtype: 3,
        torch.ops.aten.to.prim_other: 2,
        torch.ops.aten.to.device: 4,
        torch.ops.aten.to.prim_Device: 4,
    }.get(node.target if node.op == "call_function" else None)
    if copy_arg_index is not None and len(node.args) > copy_arg_index:
        return bool(node.args[copy_arg_index])

    return False


def _has_layout_change_request(node: Node, input_node: Node) -> bool:
    """Return True when the node requests a layout/memory_format/pin_memory change."""
    memory_format = node.kwargs.get("memory_format", None)

    # aten.to.dtype/other(..., memory_format)
    if node.op == "call_function" and node.target in {
        torch.ops.aten.to.dtype,
        torch.ops.aten.to.other,
    }:
        if len(node.args) > 4:
            memory_format = node.args[4]

    # aten.to.device(..., copy=False, memory_format=None)
    if node.op == "call_function" and node.target is torch.ops.aten.to.device:
        if len(node.args) > 5:
            memory_format = node.args[5]

    if node.op == "call_function" and node.target is torch.ops.aten.to.dtype_layout:
        layout = node.kwargs.get("layout", None)
        if layout is not None and layout != _layout_of(input_node):
            return True
        if node.kwargs.get("pin_memory", None):
            return True

    return memory_format not in (None, torch.preserve_format)


def _input_node_for_cast(node: Node) -> Node | None:
    if not node.args or not isinstance(node.args[0], Node):
        return None
    return node.args[0]


def _target_dtype_for_to_method(node: Node) -> torch.dtype | None:
    """Resolve the requested dtype from a Tensor.to call_method node."""
    dtype = node.kwargs.get("dtype", None)
    if dtype is not None:
        return _dtype_of(dtype)

    if len(node.args) < 2:
        return None

    target = node.args[1]
    if isinstance(target, torch.dtype):
        return target

    if _is_tensor_like_node(target):
        return _dtype_of(target)

    # torch.Tensor.to(device=None, dtype=None, ...)
    if _as_device(target) is not None:
        if len(node.args) > 2:
            return _dtype_of(node.args[2])
        return _dtype_of(node.args[0])

    return _dtype_of(target)


def _target_device_for_to_method(node: Node) -> torch.device | None:
    """Resolve the requested device from a Tensor.to call_method node."""
    device = node.kwargs.get("device", None)
    if device is not None:
        return _as_device(device)

    if len(node.args) < 2:
        return None

    target = node.args[1]
    if _is_tensor_like_node(target):
        return _device_of(target)
    device = _as_device(target)
    if device is not None:
        return device
    return None


def _target_dtype_for_function(node: Node) -> torch.dtype | None:
    """Resolve the requested dtype from a cast-like call_function node."""
    if node.target is torch.ops.prims.convert_element_type.default:
        return _dtype_of(_kw_or_arg(node, "dtype", 1))

    if node.target is torch.ops.aten.to.dtype:
        return _dtype_of(_kw_or_arg(node, "dtype", 1))

    if node.target is torch.ops.aten.to.device:
        return _dtype_of(_kw_or_arg(node, "dtype", 2))

    if node.target is torch.ops.aten.to.other:
        return _dtype_of(_kw_or_arg(node, "other", 1))

    if node.target is torch.ops.aten.to.dtype_layout:
        return _dtype_of(node.kwargs.get("dtype", None))

    if node.target is torch.ops.aten.to.prim_Device:
        return _dtype_of(_kw_or_arg(node, "dtype", 2))

    if node.target is torch.ops.aten.to.prim_dtype:
        return _dtype_of(_kw_or_arg(node, "dtype", 1))

    return None


def _target_device_for_function(node: Node) -> torch.device | None:
    if node.target is torch.ops.aten.to.device:
        return _device_of(_kw_or_arg(node, "device", 1))
    if node.target is torch.ops.aten.to.other:
        return _device_of(_kw_or_arg(node, "other", 1))
    if node.target is torch.ops.aten.to.dtype_layout:
        return _as_device(node.kwargs.get("device", None))
    if node.target is torch.ops.aten.to.prim_Device:
        return _device_of(_kw_or_arg(node, "device", 1))
    return None


def _target_dtype(node: Node) -> torch.dtype | None:
    if node.op == "call_method":
        if node.target == "to":
            return _target_dtype_for_to_method(node)
        return _DTYPE_METHOD_TARGETS.get(node.target)

    if node.op == "call_function":
        return _target_dtype_for_function(node)

    return None


def _target_device(node: Node) -> torch.device | None:
    if node.op == "call_method" and node.target == "to":
        return _target_device_for_to_method(node)
    if node.op == "call_function":
        return _target_device_for_function(node)
    return None


def _is_cast_like_node(node: Node) -> bool:
    if node.op == "call_method":
        return node.target == "to" or node.target in _DTYPE_METHOD_TARGETS

    if node.op != "call_function":
        return False

    return node.target in _CAST_FUNCTION_TARGETS


def _is_noop_cast(node: Node) -> bool:
    """Return True when the node is a cast that can be safely replaced by its input."""
    if not _is_cast_like_node(node):
        return False
    input_node = _input_node_for_cast(node)
    if input_node is None:
        return False
    if _has_explicit_copy(node) or _has_layout_change_request(node, input_node):
        return False

    input_dtype = _dtype_of(input_node)
    output_dtype = _dtype_of(node)
    requested_dtype = _target_dtype(node)

    if input_dtype is None or output_dtype is None:
        return False
    if input_dtype != output_dtype:
        return False
    if requested_dtype is not None and requested_dtype != input_dtype:
        return False

    input_device = _device_of(input_node)
    output_device = _device_of(node)
    requested_device = _target_device(node)

    if input_device is not None and output_device is not None and input_device != output_device:
        return False
    if requested_device is not None and input_device is not None and requested_device != input_device:
        return False

    return True


def eliminate_noop_casts_(gm: GraphModule) -> None:
    """
    Replace no-op cast-like nodes with their tensor input.
    """
    if os.environ.get(_DISABLE_CAST_ELIMINATION_ENV, "") == "1":
        return

    graph = gm.graph
    changed = False

    for node in list(graph.nodes):
        if not _is_noop_cast(node):
            continue

        input_node = _input_node_for_cast(node)
        if input_node is None:
            continue

        node.replace_all_uses_with(input_node)
        graph.erase_node(node)
        changed = True

    if changed:
        graph.lint()
        gm.recompile()
