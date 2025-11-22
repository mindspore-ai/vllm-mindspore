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

from typing import Any, Dict, List, Optional

from mrt.ir import GraphExecutor, Node, Op, Tensor, Value, Tuple, DataType


# ===== MLIR Type Conversion Utilities =====

def _create_tensor_value_from_mlir(mlir_tensor) -> Value:
    """Create Tensor Value from MLIR RankedTensorType."""
    shape = [int(d) if isinstance(d, int) else -1 for d in mlir_tensor.shape]
    data_type = DataType.from_string(str(mlir_tensor.element_type))
    tensor = Tensor(shape, data_type)
    return Value(tensor)


# ===== Constant Operation Handling =====

def _is_constant_op(op_name: str) -> bool:
    """Check if an operation is a constant operation."""
    return op_name in ("arith.constant", "std.constant") or op_name.startswith("mrt.constant.")

def _get_value_from_attr(value_attr):
    """Extract raw value from value_attr."""
    if hasattr(value_attr, "value"):
        return Value(value_attr.value)
    return Value(Tuple([_get_value_from_attr(item) for item in value_attr]))

def _extract_mrt_constant_value(op):
    """Extract constant value from mrt.constant.* operation."""
    if not hasattr(op, "attributes") or "value" not in op.attributes:
        raise ValueError(f"mrt.constant operation missing 'value' attribute: {op}")

    op_name = getattr(op, "name", "") or getattr(op, "OPERATION_NAME", "")
    value_attr = op.attributes["value"]
    try:
        if op_name == "mrt.constant.dtype":
            # Special handling for mrt.constant.dtype: extract type and convert to string value
            return Value(DataType.convert_str_to_int(str(value_attr.value)))
        return _get_value_from_attr(value_attr)
    except Exception as e:
        raise ValueError(f"Unable to extract value from constant operation. "
                         f"Attribute type: {type(value_attr)}, Attribute value: {value_attr}") from e

def _extract_constant_value(op):
    """Extract constant value from constant operation."""
    op_name = getattr(op, "name", "") or getattr(op, "OPERATION_NAME", "")
    if op_name.startswith("mrt.constant."):
        return _extract_mrt_constant_value(op)
    raise NotImplementedError(f"Constant operation {op_name} is not supported: {op}")

# ===== MRT Dialect Operator Mapping =====

def _map_op_name_to_runtime_op(name: str) -> Op:
    """Map MRT dialect operator name to GraphExecutor Op."""
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
    """Return function input parameters and return value operands."""
    if not func_op.regions:
        raise ValueError("func.func has no regions")

    entry_block = func_op.regions[0].blocks[0]
    inputs = list(entry_block.arguments)

    def _find_return_op(func_op):
        for region in func_op.regions:
            for blk in region.blocks:
                for op in blk.operations:
                    if getattr(op, "name", "") == "func.return":
                        return op
        return None

    ret_op = _find_return_op(func_op)
    if ret_op is None:
        raise ValueError("func.return not found in function body")
    outputs = list(ret_op.operands)
    return inputs, outputs


def _iter_ops_in_func(func_op):
    """Traverse all operations in func.func."""
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
    """Extract function symbol name from attributes."""
    if hasattr(func_op, "attributes") and "sym_name" in func_op.attributes:
        val = str(func_op.attributes["sym_name"])
        # Usually printed in '"main"' format
        return val.strip('"')
    return None

def _get_func_op(mlir_module):
    """Find func.func operation by name or select default."""
    candidates = _func_candidates(mlir_module)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    for f in candidates:
        name = _func_name_from_attr(f)
        if name in ("main", "forward"):
            return f
    return candidates[0]

# ===== Main Builder Function =====

_GLOBAL_GRAPH_ID = 0

def _next_unique_graph_id():
    global _GLOBAL_GRAPH_ID
    _GLOBAL_GRAPH_ID += 1
    return _GLOBAL_GRAPH_ID

def gen_executor_from_mlir_module(mlir_module):
    """Build callable GraphExecutor from MLIR module (MRT dialect).

    This function assumes MLIR module is already converted to MRT dialect, where
    operators have one-to-one correspondence with GraphExecutor's Op.

    Workflow:
    1. Find target func.func
    2. Create GraphExecutor instance with unique name
    3. Create value nodes for function parameters and add as parameters
    4. Traverse all operations in function:
       - Map MRT dialect operators to Op
       - Create corresponding op nodes
    5. Set return values
    6. Build and return callable executor

    Args:
        mlir_module: MLIR Module object containing MRT dialect.

    Returns:
        A tuple of (GraphExecutor, placeholder_nodes) for building executable graph.

    Raises:
        ValueError: If func.func not found in MLIR module.
        NotImplementedError: If unsupported operation encountered or operation has multiple results.
    """
    # Find target function
    func_op = _get_func_op(mlir_module)
    if func_op is None:
        raise ValueError("func.func not found in MLIR module")

    # Create GraphExecutor
    executor = GraphExecutor(f"mrt_graph_{_next_unique_graph_id()}")

    # Environment: MLIR Value -> GraphExecutor Node mapping
    env: Dict[Any, Node] = {}

    # Get function inputs and outputs
    func_inputs, func_outputs = _get_func_io(func_op)

    # Create value node for each function parameter
    input_nodes: List[Node] = []
    for arg in func_inputs:
        tensor_value = _create_tensor_value_from_mlir(arg.type)
        val_node = executor.add_value_node(tensor_value)
        env[arg] = val_node
        input_nodes.append(val_node)

    with executor:
        for val_node in input_nodes:
            executor.add_parameter(val_node)

        for op in _iter_ops_in_func(func_op):
            op_name = op.name
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
                val_node = executor.add_value_node(constant_value)
                env[results[0]] = val_node
                continue

            runtime_op = _map_op_name_to_runtime_op(op_name)

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

            out_tensor_value = _create_tensor_value_from_mlir(results[0].type)
            node = executor.add_op_node(runtime_op, input_nodes_for_op, out_tensor_value)
            env[results[0]] = node

        if len(func_outputs) > 0:
            return_nodes = [env[v] for v in func_outputs]
            executor.make_tuple(return_nodes)
        executor.set_return()

    placeholder_nodes = [env[arg] for arg in func_inputs]
    return executor, placeholder_nodes


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
