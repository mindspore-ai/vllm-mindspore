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

"""Build GraphExecutor from MLIR modules (MRT dialect).

This module provides functionality to convert MRT dialect MLIR modules into
executable GraphExecutor instances. Operators in MRT dialect have one-to-one
correspondence with GraphExecutor supported operators.
"""

import ast
from typing import Any, Dict, List, Optional

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from mrt.ir import GraphExecutor, Node, Op
from mrt.torch.utils import from_torch, to_torch, update_tensor_data


# ===== MLIR Type Conversion Utilities =====

def _elemtype_to_torch_dtype(elem_ty) -> torch.dtype:
    """Convert MLIR element type to torch.dtype.
    
    Args:
        elem_ty: MLIR element type object or string.
        
    Returns:
        Corresponding torch.dtype.
        
    Raises:
        NotImplementedError: If element type is not supported.
    """
    s = str(elem_ty)
    dtype_map = {
        "f32": torch.float32,
        "tensor<f32>": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "i64": torch.int64,
        "si64": torch.int64,
        "i32": torch.int32,
        "si32": torch.int32,
        "i16": torch.int16,
        "si16": torch.int16,
        "i8": torch.int8,
        "si8": torch.int8,
        "ui8": torch.uint8,
        "uint8": torch.uint8,
    }

    if s in dtype_map:
        return dtype_map[s]

    raise NotImplementedError(f"Unsupported element type: {elem_ty!r}")


def _ranked_tensor_type_to_shape_dtype(tensor_ty):
    """Extract shape and dtype from MLIR RankedTensorType.
    
    Args:
        tensor_ty: MLIR RankedTensorType object.
        
    Returns:
        (shape: List[int], dtype: torch.dtype) tuple.
    """
    try:
        shape = list(tensor_ty.shape)  # type: ignore[attr-defined]
        elem_ty = tensor_ty.element_type  # type: ignore[attr-defined]
    except AttributeError as e:
        # Fallback: parse from string (e.g. 'tensor<?xf32>' or 'tensor<2x3xf32>')
        ts = str(tensor_ty)
        if not (ts.startswith("tensor<") and ts.endswith(">")):
            raise ValueError(f"Unrecognized tensor type text format: {ts}") from e

        core = ts[len("tensor<"): -1]
        dims, et = core.rsplit("x", 1)
        elem_ty = et
        shape = []
        for d in dims.split("x"):
            if d == "?":
                shape.append(-1)
            else:
                shape.append(int(d))

    # Convert potential MLIR symbolic dimensions to -1
    shape = [int(d) if isinstance(d, int) else -1 for d in shape]
    dtype = _elemtype_to_torch_dtype(elem_ty)
    return shape, dtype


def _ranked_tensor_type_to_dummy(tensor_ty, is_fake: bool) -> torch.Tensor:
    """Create FakeTensor or empty Tensor with the same shape/dtype.
    
    Args:
        tensor_ty: MLIR RankedTensorType object.
        is_fake: Whether to create FakeTensor.
        
    Returns:
        torch.Tensor instance.
    """
    shape, dtype = _ranked_tensor_type_to_shape_dtype(tensor_ty)
    if is_fake:
        fake_mode = FakeTensorMode()
        return fake_mode.from_tensor(torch.empty(shape, dtype=dtype))
    return torch.empty(shape, dtype=dtype)


# ===== Constant Operation Handling =====

def _is_constant_op(op_name: str) -> bool:
    """Check if an operation is a constant operation.
    
    Common constant operations include:
    - arith.constant (standard MLIR arithmetic dialect constant)
    - std.constant (standard dialect constant, deprecated)
    - mrt.constant (MRT dialect constant, if any)
    
    Args:
        op_name: Operation name (e.g. "arith.constant").
        
    Returns:
        True if constant operation, False otherwise.
    """
    constant_ops = {
        "arith.constant",
        "std.constant",
        "mrt.constant",
    }
    return op_name in constant_ops


def _extract_constant_value(op):
    """Extract constant value from constant operation.

    This function parses MLIR constant operation attributes, extracts constant
    value and converts it to Python/Torch object.

    Args:
        op: MLIR Operation object (constant operation).

    Returns:
        Constant value (Python object or torch.Tensor).

    Raises:
        ValueError: If unable to extract constant value.

    Example:
        For `%cst = arith.constant dense<[3, 2]> : tensor<2xi64>`
        Returns torch.tensor([3, 2], dtype=torch.int64)
    """
    # Get 'value' attribute (standard attribute for arith.constant)
    if not hasattr(op, "attributes") or "value" not in op.attributes:
        raise ValueError(f"Constant operation missing 'value' attribute: {op}")

    value_attr = op.attributes["value"]

    try:
        if hasattr(value_attr, "value"):
            return value_attr.value

        if hasattr(value_attr, "type"):
            attr_type = value_attr.type

            if hasattr(value_attr, "to_array"):
                array = value_attr.to_array()
                return torch.from_numpy(array)

            value_str = str(value_attr)

            if "dense<" in value_str:
                start = value_str.index("dense<") + 6
                end = value_str.index(">", start)
                content = value_str[start:end].strip()

                type_str = str(attr_type)
                elem_type = (attr_type.element_type if hasattr(attr_type, 'element_type')
                            else type_str)
                dtype = _elemtype_to_torch_dtype(elem_type)

                if content.startswith("[") and content.endswith("]"):
                    values = ast.literal_eval(content)
                    if not isinstance(values, (list, tuple)):
                        values = [values]
                    return torch.tensor(values, dtype=dtype)
                value = ast.literal_eval(content)
                return torch.tensor(value, dtype=dtype)

            if "unit" in value_str:
                return None

            try:
                parsed_value = ast.literal_eval(value_str)
                return torch.tensor(parsed_value)
            except (ValueError, SyntaxError):
                pass

        raise ValueError(
            f"Unable to extract value from constant operation. "
            f"Attribute type: {type(value_attr)}, Attribute value: {value_attr}"
        )

    except Exception as e:
        raise ValueError(
            f"Error extracting constant value: {e}\n"
            f"Operation: {op}\n"
            f"Value attribute: {value_attr}"
        ) from e


# ===== MRT Dialect Operator Mapping =====

def _map_mrt_op_name_to_runtime_op(name: str) -> Op:
    """Map MRT dialect operator name to GraphExecutor Op.
    
    Operators in MRT dialect have one-to-one correspondence with GraphExecutor
    supported Ops. This function parses operator name (removes dialect prefix)
    and looks up corresponding Op enum directly.
    
    Args:
        name: MLIR operator name (dialect.op format, e.g. "mrt.add").
        
    Returns:
        Corresponding Op enum value.
        
    Raises:
        NotImplementedError: If operator is not supported.
    
    Example:
        >>> _map_mrt_op_name_to_runtime_op("mrt.add")  # returns Op.add
        >>> _map_mrt_op_name_to_runtime_op("mrt.matmul")  # returns Op.matmul
    """
    # Extract operator name from full name (remove dialect prefix)
    # e.g.: "mrt.add" -> "add"
    if "." in name:
        _, op_name = name.rsplit(".", 1)
    else:
        op_name = name

    # Look up Op enum directly by attribute name
    if hasattr(Op, op_name):
        return getattr(Op, op_name)

    # If not found, raise error
    raise NotImplementedError(
        f"Unsupported operator: {name} "
        f"('{op_name}' not found in Op enum)"
    )


# ===== MLIR Module Traversal Utilities =====

def _top_module_op(module) -> Any:
    """Return top-level Operation of MLIR Module."""
    return module.operation


def _get_func_io(func_op):
    """Return function input parameters (block arguments) and return value operands.

    Args:
        func_op: func.func Operation.

    Returns:
        (inputs: List[Any], outputs: List[Any]) tuple.
    """
    if not func_op.regions:
        raise ValueError("func.func has no regions")

    entry_block = func_op.regions[0].blocks[0]
    inputs = list(entry_block.arguments)

    # Find func.return
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
    """Traverse all operations in func.func in block order."""
    for region in func_op.regions:
        for blk in region.blocks:
            yield from blk.operations


def _func_candidates(mlir_module) -> List[Any]:
    """Collect candidate func.func operations from module."""
    top = _top_module_op(mlir_module)
    candidates: List[Any] = []
    for region in top.regions:
        for block in region.blocks:
            for op in block.operations:
                if getattr(op, "OPERATION_NAME", "") == "func.func":
                    candidates.append(op)
    return candidates


def _func_name_from_attr(func_op) -> Optional[str]:
    """Extract function symbol name from attributes (if exists)."""
    if hasattr(func_op, "attributes") and "sym_name" in func_op.attributes:
        val = str(func_op.attributes["sym_name"])
        # Usually printed in '"main"' format
        return val.strip('"')
    return None


def _pick_default_func(candidates: List[Any]) -> Optional[Any]:
    """Select default function, preferring 'main' or 'forward'."""
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
    """Find func.func operation by optional name, otherwise select default function."""
    candidates = _func_candidates(mlir_module)
    if func_name is None:
        return _pick_default_func(candidates)

    for f in candidates:
        name = _func_name_from_attr(f)
        if name == func_name:
            return f
        # Some bindings preserve quotes in str(); tolerate exact quoted form
        if name is None and hasattr(f, "attributes") and "sym_name" in f.attributes:
            if str(f.attributes["sym_name"]) == f'"{func_name}"':
                return f
    return None


# ===== Main Builder Function =====

def build_executor_from_mlir_module(
    mlir_module,
    func_name: Optional[str] = None
):
    """Build callable GraphExecutor from MLIR module (MRT dialect).

    This function assumes MLIR module is already converted to MRT dialect, where
    operators have one-to-one correspondence with GraphExecutor's Op.

    Workflow:
    1. Find target func.func (defaults to 'main' or 'forward')
    2. Create GraphExecutor instance
    3. Create value nodes for function parameters and add as parameters
    4. Traverse all operations in function:
       - Map MRT dialect operators to Op
       - Create corresponding op nodes
    5. Set return values
    6. Build and return callable executor

    Args:
        mlir_module: MLIR Module object containing MRT dialect.
        func_name: Optional function name to select specific function.
                  If None, selects 'main', 'forward', or first function.

    Returns:
        A callable object accepting torch.Tensor inputs and returning torch objects.

    Raises:
        ValueError: If specified function not found.
        NotImplementedError: If unsupported operator encountered.

    Example:
        >>> mlir_module = parse_mlir_module_from_text(mrt_dialect_text)
        >>> executor_fn = build_executor_from_mlir_module(mlir_module)
        >>> result = executor_fn(input_tensor1, input_tensor2)
    """
    # Find target function
    func_op = _get_func_op(mlir_module, func_name)
    if func_op is None:
        raise ValueError(
            f"func.func @{func_name if func_name else 'default'} not found"
        )

    # Create GraphExecutor
    graph_name = func_name if func_name else "mrt_graph_exec"
    executor = GraphExecutor(graph_name)

    # Environment: MLIR Value -> GraphExecutor Node mapping
    env: Dict[Any, Node] = {}

    # Get function inputs and outputs
    func_inputs, func_outputs = _get_func_io(func_op)

    # Create value node for each function parameter
    input_nodes: List[Node] = []
    for arg in func_inputs:
        arg_ty = getattr(arg, "type")
        dummy = _ranked_tensor_type_to_dummy(arg_ty, is_fake=False)
        val_node = executor.add_value_node(from_torch(dummy))
        env[arg] = val_node
        input_nodes.append(val_node)

    with executor:
        for val_node in input_nodes:
            executor.add_parameter(val_node)

        for op in _iter_ops_in_func(func_op):
            op_name = getattr(op, "name", "")

            if op_name == "func.return":
                continue

            if _is_constant_op(op_name):
                constant_value = _extract_constant_value(op)
                results = list(op.results)
                if len(results) != 1:
                    raise NotImplementedError(
                        f"Constant operation {op_name} has {len(results)} results; "
                        f"currently only single-result supported"
                    )
                val_node = executor.add_value_node(from_torch(constant_value))
                env[results[0]] = val_node
                continue

            runtime_op = _map_mrt_op_name_to_runtime_op(op_name)

            input_nodes_for_op: List[Node] = []
            for operand in op.operands:
                if operand not in env:
                    raise RuntimeError(f"Operand for operator {op_name} not ready")
                input_nodes_for_op.append(env[operand])

            results = list(op.results)
            if len(results) != 1:
                raise NotImplementedError(
                    f"{op_name} has {len(results)} results; "
                    f"currently only single-result operators supported"
                )

            out_ty = results[0].type
            out_dummy = _ranked_tensor_type_to_dummy(out_ty, is_fake=True)

            node = executor.add_op_node(runtime_op, input_nodes_for_op, from_torch(out_dummy))
            env[results[0]] = node

        if len(func_outputs) == 0:
            executor.set_return()
        elif len(func_outputs) == 1:
            return_nodes = [env[v] for v in func_outputs]
            executor.make_tuple(return_nodes)
            executor.set_return()
        else:
            return_nodes = [env[v] for v in func_outputs]
            executor.make_tuple(return_nodes)
            executor.set_return()

    executor.dump_graph()
    executor.build()

    placeholder_nodes = [env[arg] for arg in func_inputs]

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


# ===== Additional Helper Functions =====

def print_mlir_module_structure(mlir_module):
    """Print MLIR module structure information (for debugging).

    Args:
        mlir_module: MLIR Module object.
    """
    print("=" * 80)
    print("MLIR Module Structure:")
    print("=" * 80)

    candidates = _func_candidates(mlir_module)
    print(f"Found {len(candidates)} function(s):")

    for i, func_op in enumerate(candidates):
        func_name = _func_name_from_attr(func_op)
        print(f"\nFunction {i + 1}: {func_name or '(unnamed)'}")

        try:
            inputs, outputs = _get_func_io(func_op)
            print(f"  Inputs: {len(inputs)}")
            for j, inp in enumerate(inputs):
                inp_ty = getattr(inp, "type")
                print(f"    {j}: {inp_ty}")

            print(f"  Outputs: {len(outputs)}")
            for j, out in enumerate(outputs):
                out_ty = getattr(out, "type")
                print(f"    {j}: {out_ty}")

            ops = list(_iter_ops_in_func(func_op))
            print(f"  Operations: {len(ops)}")
            op_names = {}
            for op in ops:
                name = getattr(op, "name", "")
                op_names[name] = op_names.get(name, 0) + 1

            for name, count in sorted(op_names.items()):
                print(f"    {name}: {count}")

        except Exception as e:
            print(f"  Error: {e}")

    print("=" * 80)
