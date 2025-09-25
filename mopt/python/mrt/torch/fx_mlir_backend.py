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
- Helpers to parse MLIR modules/functions and build a GraphExecutor.
- A small runtime op mapping from MLIR op names to executor ops.
- An FX backend entry that applies decompositions and imports to StableHLO.
"""

from typing import Any, Dict, List, Mapping, Optional, Tuple as PyTuple

import torch
from torch._decomp import get_decompositions
from torch.fx.experimental.proxy_tensor import make_fx
from torch.func import functionalize
from torch._subclasses.fake_tensor import FakeTensorMode

from mrt.ir import GraphExecutor, Node, Op
from mrt.torch.utils import from_torch, to_torch, update_tensor_data
from mrt.torch.decompositions import DEFAULT_DECOMPOSITIONS


def _elemtype_to_torch_dtype(elem_ty) -> torch.dtype:
    """Map MLIR element type to torch.dtype.

    Args:
        elem_ty: MLIR element type object or string-like.

    Returns:
        torch.dtype corresponding to elem_ty.

    Raises:
        NotImplementedError: If the element type is unsupported.
    """
    # Typical bindings: elem_ty is a subclass of mlir.ir.Type; using str(elem_ty)
    s = str(elem_ty)
    if s in ("f32", "tensor<f32>"):
        return torch.float32
    if s == "f16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    if s in ("i64", "si64"):
        return torch.int64
    if s in ("i32", "si32"):
        return torch.int32
    if s in ("i16", "si16"):
        return torch.int16
    if s in ("i8", "si8"):
        return torch.int8
    if s in ("ui8", "uint8"):
        return torch.uint8
    raise NotImplementedError(f"Unsupported element type: {elem_ty!r}")


def _ranked_tensor_type_to_shape_dtype(tensor_ty):
    """Extract shape and dtype from an MLIR RankedTensorType-like object.

    Supports common bindings exposing attributes:
    - tensor_ty.shape: tuple[int | Attribute]
    - tensor_ty.element_type

    Falls back to parsing textual form like 'tensor<?xf32>'.

    Returns:
        A pair (shape: List[int], dtype: torch.dtype)
    """
    # Adapt to mlir.ir.RankedTensorType
    # Common interface: tensor_ty.shape -> tuple[int|mlir.ir.Attribute], tensor_ty.element_type
    try:
        shape = list(tensor_ty.shape)  # type: ignore[attr-defined]
        elem_ty = tensor_ty.element_type  # type: ignore[attr-defined]
    except AttributeError:
        # Fallback: parse via string (covers textual MLIR)
        # Forms like: 'tensor<?xf32>' or 'tensor<2x3xf32>'
        ts = str(tensor_ty)  # e.g., 'tensor<?xf32>'
        if not (ts.startswith("tensor<") and ts.endswith(">")):
            # Provide clearer error than a ValueError deep in parsing
            raise ValueError(f"Unrecognized tensor type textual form: {ts}")
        core = ts[len("tensor<"): -1]
        dims, et = core.rsplit("x", 1)
        elem_ty = et
        shape = []
        for d in dims.split("x"):
            if d == "?":
                shape.append(-1)
            else:
                shape.append(int(d))
    # Convert possible MLIR symbolic dims to -1
    shape = [int(d) if isinstance(d, int) else -1 for d in shape]
    dtype = _elemtype_to_torch_dtype(elem_ty)
    return shape, dtype


def _ranked_tensor_type_to_dummy(tensor_ty, is_fake) -> torch.Tensor:
    """Create a FakeTensor with the same shape/dtype as the MLIR tensor type."""
    shape, dtype = _ranked_tensor_type_to_shape_dtype(tensor_ty)
    if is_fake:
        fake_mode = FakeTensorMode()
        return fake_mode.from_tensor(torch.empty(shape, dtype=dtype))
    return torch.empty(shape, dtype=dtype)


def _map_mlir_op_name_to_runtime_op(name: str) -> Op:
    """Map MLIR op name (dialect.op) to runtime Op enum/class used by GraphExecutor.

    Raises:
        NotImplementedError: if the op name is not supported.
    """
    m = {
        "tosa.add": Op.add,
        "tosa.sub": Op.sub,
        "tosa.mul": Op.mul,
        "tosa.matmul": Op.matmul,
        "tosa.reshape": Op.reshape,
        "tosa.transpose": Op.transpose,
        "tosa.concat": Op.concat,
        "tosa.negate": Op.neg,
        "tosa.square": Op.square,
        "tosa.rsqrt": Op.rsqrt,
        "tosa.relu": Op.relu,
        "tosa.sigmoid": Op.sigmoid,
        "tosa.softmax": Op.softmax,
        "stablehlo.dot_general": Op.matmul,
        "stablehlo.reshape": Op.reshape,
    }
    if name not in m:
        raise NotImplementedError(f"Unsupported op: {name}")
    return m[name]


# ===== Module/function traversal (based on module.operation) =====


def _top_module_op(module) -> Any:
    """Return the top-level Operation from an mlir.ir.Module-like object."""
    # module is mlir.ir.Module
    # Get its top-level operation for traversal
    return module.operation


def _get_func_io(func_op) -> PyTuple[List[Any], List[Any]]:
    """Return function inputs (block arguments) and return operands."""
    # func_op is an Operation (func.func)
    # Its body is typically at func_op.regions[0].blocks[0]
    if not func_op.regions:
        raise ValueError("func.func has no regions")
    entry_block = func_op.regions[0].blocks[0]
    inputs = list(entry_block.arguments)

    # Find func.return (possibly in the same block)
    ret_op = None
    for op in entry_block.operations:
        if getattr(op, "name", "") == "func.return":
            ret_op = op
            break
    if ret_op is None:
        # Traverse all blocks
        for region in func_op.regions:
            for blk in region.blocks:
                for op in blk.operations:
                    if getattr(op, "name", "") == "func.return":
                        ret_op = op
                        break
                if ret_op:
                    break
            if ret_op:
                break
    if ret_op is None:
        raise ValueError("func.return not found in function body")

    outputs = list(ret_op.operands)
    return inputs, outputs


def _iter_ops_in_func(func_op):
    """Yield all operations in a func.func in block order."""
    for region in func_op.regions:
        for blk in region.blocks:
            for op in blk.operations:
                yield op


def _func_candidates(mlir_module) -> List[Any]:
    """Collect candidate func.func operations from a module."""
    top = _top_module_op(mlir_module)
    candidates: List[Any] = []
    for region in top.regions:
        for block in region.blocks:
            for op in block.operations:
                if getattr(op, "OPERATION_NAME", "") == "func.func":
                    candidates.append(op)
    return candidates


def _func_name_from_attr(func_op) -> Optional[str]:
    """Extract function symbol name from attributes if present."""
    if hasattr(func_op, "attributes") and "sym_name" in func_op.attributes:
        val = str(func_op.attributes["sym_name"])
        # Typically printed as '"main"'
        return val.strip('"')
    return None


def _pick_default_func(candidates: List[Any]) -> Optional[Any]:
    """Pick a default function, preferring 'main' or 'forward'."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    for f in candidates:
        name = _func_name_from_attr(f)
        if name in ("main", "forward"):
            return f
    return candidates[0]


def _get_func_op(mlir_module, func_name: Optional[str] = None):
    """Find a func.func operation by optional name, else pick a default."""
    candidates = _func_candidates(mlir_module)
    if func_name is None:
        return _pick_default_func(candidates)

    for f in candidates:
        name = _func_name_from_attr(f)
        if name == func_name:
            return f
        # Some bindings keep the quotes in str(); tolerate exact quoted form
        if name is None and hasattr(f, "attributes") and "sym_name" in f.attributes:
            if str(f.attributes["sym_name"]) == f'"{func_name}"':
                return f
    return None


# ===== Main build function =====


def build_executor_from_mlir_module(mlir_module, func_name: Optional[str] = None):
    """Build a callable executor from an MLIR module.

    Args:
        mlir_module: An mlir.ir.Module-like object.
        func_name: Optional symbol name to select a specific function.

    Returns:
        A Python callable that accepts torch.Tensor inputs and returns a torch object.
    """
    # Find func.func at top-level; if func_name is specified, match the symbol name
    func_op = _get_func_op(mlir_module, func_name)
    if func_op is None:
        raise ValueError(f"func.func @{func_name} not found")

    executor = GraphExecutor("mlir_graph_exec")
    env: Dict[Any, Node] = {}

    func_inputs, func_outputs = _get_func_io(func_op)
    params: List[Node] = []
    for arg in func_inputs:
        arg_ty = getattr(arg, "type")
        dummy = _ranked_tensor_type_to_dummy(arg_ty, False)
        val_node = executor.add_value_node(from_torch(dummy))
        env[arg] = val_node
        params.append(val_node)
    with executor:
        # 1) Parameters
        for val_node in params:
            executor.add_parameter(val_node)

        # 2) Traverse ops
        for op in _iter_ops_in_func(func_op):
            name = getattr(op, "name", "")

            if name == "func.return":
                continue

            # Regular ops (single result)
            runtime_op = _map_mlir_op_name_to_runtime_op(name)

            # Operand nodes
            inputs: List[Node] = []
            for operand in op.operands:
                if operand not in env:
                    raise RuntimeError(f"Operand not ready for {name}")
                inputs.append(env[operand])

            results = list(op.results)
            if len(results) != 1:
                raise NotImplementedError(
                    f"{name} has {len(results)} results; only single-result ops supported"
                )

            out_ty = results[0].type
            out_dummy = _ranked_tensor_type_to_dummy(out_ty, True)

            node = executor.add_op_node(runtime_op, inputs, from_torch(out_dummy))
            env[results[0]] = node

        # 3) Return
        if not func_outputs:
            executor.set_return()
        else:
            # Multiple returns: aggregate and then return
            executor.make_tuple([env[v] for v in func_outputs])
            executor.set_return()

    executor.dump_graph()
    executor.build()

    # Record parameter nodes in input order
    placeholder_nodes = [env[a] for a in func_inputs]

    def compiled_callable(*new_inputs: torch.Tensor):
        """Run the compiled executor with new torch.Tensor inputs."""
        if len(new_inputs) != len(placeholder_nodes):
            raise ValueError(
                f"Expected {len(placeholder_nodes)} inputs, but got {len(new_inputs)}"
            )
        for i, p_node in enumerate(placeholder_nodes):
            if p_node.output.is_tensor():
                update_tensor_data(p_node.output.to_tensor(), new_inputs[i])
            else:
                p_node.output = from_torch(new_inputs[i])
        result = executor.run()
        return to_torch(result)

    return compiled_callable


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
    """FX backend entry point: decompose, import to StableHLO, and build executor."""
    from torch_mlir import fx  # grouped third-party import kept local to reduce module import overhead
    from torch_mlir.compiler_utils import OutputType

    gm = apply_decompositions(gm, example_inputs, DEFAULT_DECOMPOSITIONS)

    mlir_module = fx.stateless_fx_import(gm, output_type=OutputType.STABLEHLO)

    return build_executor_from_mlir_module(mlir_module)
