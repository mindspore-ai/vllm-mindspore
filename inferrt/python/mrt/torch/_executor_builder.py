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

import torch
from mopt import ir
from mrt.ir import (
    GraphExecutor,
    Node,
    Op,
    Tensor,
    Value,
    Tuple,
    DataType,
    Device,
    DeviceType,
)
from mrt.ir import SymbolicVar, SymbolicConst, SymbolicExpr

# ===== MLIR Type Conversion Utilities =====


def _mlir_device_to_device(mlir_device) -> Device:
    """Convert MLIR device to Device enum."""
    if mlir_device is None:
        return Device(DeviceType.CPU, -1)
    mlir_device_type_map = {
        "npu": DeviceType.NPU,
        "cpu": DeviceType.CPU,
    }
    device_type = mlir_device_type_map.get(mlir_device.device_type)
    if device_type is None:
        raise NotImplementedError(f"Unsupported device: {mlir_device.device_type}")
    return Device(device_type, mlir_device.index)


def _create_mrt_value_from_mlir_type(mlir_type) -> Value:
    """Create MRT Value from MLIR Type."""
    if str(mlir_type) == "!mrt.i64":
        return Value(-1)
    if str(mlir_type).startswith("!mrt.list<"):
        return Value(Tuple([]))
    shape = [int(d) if int(d) >= 0 else -1 for d in mlir_type.shape]
    data_type = DataType.from_string(str(mlir_type.element_type))
    device = _mlir_device_to_device(mlir_type.device)
    tensor = Tensor(shape, data_type, device)
    return Value(tensor)


# ===== Constant Operation Handling =====


def _is_constant_op(op_name: str) -> bool:
    """Check if an operation is a constant operation."""
    return op_name in ("arith.constant", "std.constant") or op_name.startswith(
        "mrt.constant."
    )


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
        raise ValueError(
            f"Unable to extract value from constant operation. "
            f"Attribute type: {type(value_attr)}, Attribute value: {value_attr}"
        ) from e


def _extract_constant_value(op):
    """Extract constant value from constant operation."""
    op_name = getattr(op, "name", "") or getattr(op, "OPERATION_NAME", "")
    if op_name == "mrt.constant.none":
        return Value()
    if op_name.startswith("mrt.constant."):
        return _extract_mrt_constant_value(op)
    raise NotImplementedError(f"Constant operation {op_name} is not supported: {op}")


# ===== MRT Dialect Operator Mapping =====


def _map_op_name_to_runtime_op(name: str) -> Op:
    """Map MRT dialect operator name to GraphExecutor Op."""
    if name == "mrt.make_list":
        return Op.make_tuple

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
        f"Unsupported operator: {name} ('{op_name}' not found in Op enum)"
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


def _convert_affine_expr(
    expr: ir.AffineExpr, symbol_vals: List[SymbolicExpr]
) -> List[SymbolicExpr]:
    """Convert MLIR affine expression to SymbolicExpr."""
    if ir.AffineSymbolExpr.isinstance(expr):
        expr = ir.AffineSymbolExpr(expr)
        return symbol_vals[expr.position]

    if ir.AffineConstantExpr.isinstance(expr):
        expr = ir.AffineConstantExpr(expr)
        return SymbolicConst(expr.value)

    if ir.AffineAddExpr.isinstance(expr):
        expr = ir.AffineAddExpr(expr)
        lhs = _convert_affine_expr(expr.lhs, symbol_vals)
        rhs = _convert_affine_expr(expr.rhs, symbol_vals)
        return lhs + rhs

    if ir.AffineMulExpr.isinstance(expr):
        expr = ir.AffineMulExpr(expr)
        lhs = _convert_affine_expr(expr.lhs, symbol_vals)
        rhs = _convert_affine_expr(expr.rhs, symbol_vals)
        return lhs * rhs

    if ir.AffineFloorDivExpr.isinstance(expr):
        expr = ir.AffineFloorDivExpr(expr)
        lhs = _convert_affine_expr(expr.lhs, symbol_vals)
        rhs = _convert_affine_expr(expr.rhs, symbol_vals)
        return lhs // rhs

    if ir.AffineCeilDivExpr.isinstance(expr):
        expr = ir.AffineCeilDivExpr(expr)
        lhs = _convert_affine_expr(expr.lhs, symbol_vals)
        rhs = _convert_affine_expr(expr.rhs, symbol_vals)
        return lhs.__ceildiv__(rhs)

    if ir.AffineModExpr.isinstance(expr):
        expr = ir.AffineModExpr(expr)
        lhs = _convert_affine_expr(expr.lhs, symbol_vals)
        rhs = _convert_affine_expr(expr.rhs, symbol_vals)
        return lhs % rhs

    raise NotImplementedError(f"Unsupported affine expr: {expr},  type: {type(expr)}")


class ExecutorBuilder:
    """Helper class to build GraphExecutor from MLIR function."""

    def __init__(self, graph_name: Optional[str] = None):
        """Initialize ExecutorBuilder."""
        if graph_name is None:
            graph_name = f"mrt_graph_{_next_unique_graph_id()}"
        self.executor = GraphExecutor(graph_name)
        self.env: Dict[Any, Node] = {}
        self.symbol_map: Dict[str, SymbolicExpr] = {}
        self.symbol_env: Dict[Any, SymbolicExpr] = {}

    def build(self, mlir_module, fake_inputs):
        """Build GraphExecutor from MLIR module."""
        # Find target function
        func_op = _get_func_op(mlir_module)
        if func_op is None:
            raise ValueError("func.func not found in MLIR module")

        # Get function inputs and outputs
        func_inputs, func_outputs = _get_func_io(func_op)

        # Create value node for each function parameter
        input_nodes = self._create_input_nodes(func_inputs, fake_inputs)

        with self.executor:
            for val_node in input_nodes:
                self.executor.add_parameter(val_node)

            self._process_ops(func_op)

            if len(func_outputs) > 0:
                return_nodes = [self.env[v] for v in func_outputs]
                self.executor.make_tuple(return_nodes)
            self.executor.set_return()

        placeholder_nodes = [self.env[arg] for arg in func_inputs]
        return self.executor, placeholder_nodes

    def _create_input_nodes(self, func_inputs, fake_inputs) -> List[Node]:
        """Create input nodes for function parameters."""
        input_nodes = []
        for arg, fake_input in zip(func_inputs, fake_inputs):
            if isinstance(fake_input, torch.SymInt):
                sym_name = str(fake_input.node.expr)
                self.symbol_map[sym_name] = SymbolicVar(sym_name)
                mrt_value = Value(self.symbol_map[sym_name])
            else:
                mrt_value = _create_mrt_value_from_mlir_type(arg.type)
            val_node = self.executor.add_value_node(mrt_value)
            self.env[arg] = val_node
            input_nodes.append(val_node)
        return input_nodes

    def _process_ops(self, func_op):
        """Process all operations in the function."""
        for op in _iter_ops_in_func(func_op):
            self._dispatch_op(op)

    def _dispatch_op(self, op):
        """Dispatch operation to appropriate handler."""
        op_name = op.name
        if op_name == "func.return":
            return

        if op_name == "mrt.symbolic_int":
            self._handle_symbolic_int(op)
            return

        if op_name == "mrt.bind_symbolic_shape":
            self._handle_bind_symbolic_shape(op)
            return

        if _is_constant_op(op_name):
            self._handle_constant(op, op_name)
            return

        if op_name == "mrt.unpack_list":
            self._handle_unpack_list(op)
            return

        self._handle_runtime_op(op, op_name)

    def _handle_symbolic_int(self, op):
        """Handle mrt.symbolic_int operation."""
        sym_name = ir.StringAttr(op.attributes["symbol_name"]).value
        self.symbol_env[op.result] = self.symbol_map[sym_name]

    def _handle_bind_symbolic_shape(self, op):
        """Handle mrt.bind_symbolic_shape operation."""
        target = op.operands[0]
        shape_symbols = list(op.operands)[1:]
        affine_map = ir.AffineMapAttr(op.attributes["shape_expressions"]).value

        symbol_vals = [self.symbol_env[s] for s in shape_symbols]
        symbolic_shape = [
            _convert_affine_expr(expr, symbol_vals)
            for expr in affine_map.results
        ]

        self.env[target].output.to_tensor().symbolic_shape = symbolic_shape

    def _handle_constant(self, op, op_name):
        """Handle constant operations."""
        constant_value = _extract_constant_value(op)
        results = list(op.results)
        if len(results) != 1:
            raise NotImplementedError(
                f"Constant operation {op_name} has {len(results)} results; "
                f"currently only single-result supported"
            )
        val_node = self.executor.add_value_node(constant_value)
        self.env[results[0]] = val_node

    def _handle_unpack_list(self, op):
        """Handle mrt.unpack_list operation."""
        input_nodes_for_op = self._get_input_nodes(op)
        results = list(op.results)

        result_values = []
        for result in results:
            mrt_value = _create_mrt_value_from_mlir_type(result.type)
            result_values.append(mrt_value)

        # update the output value of input node with current results
        tuple_value = Value(Tuple(result_values))
        node = input_nodes_for_op[0]
        node.output = tuple_value

        for i, result in enumerate(results):
            self.env[result] = self.executor.add_op_node(
                Op.tuple_getitem,
                [node, self.executor.add_value_node(Value(i))],
                result_values[i],
            )

    def _handle_runtime_op(self, op, op_name):
        """Handle runtime operations."""
        input_nodes_for_op = self._get_input_nodes(op)
        results = list(op.results)
        runtime_op = _map_op_name_to_runtime_op(op_name)

        if runtime_op == Op.make_tuple:
            self.env[results[0]] = self.executor.make_tuple(input_nodes_for_op)
            return

        if len(results) == 1:
            mrt_value = _create_mrt_value_from_mlir_type(results[0].type)
            node = self.executor.add_op_node(runtime_op, input_nodes_for_op, mrt_value)
            self.env[results[0]] = node
        else:
            self._handle_multi_result_op(runtime_op, input_nodes_for_op, results)

    def _handle_multi_result_op(self, runtime_op, input_nodes, results):
        """Handle operations with multiple results."""
        result_values = []
        for result in results:
            mrt_value = _create_mrt_value_from_mlir_type(result.type)
            result_values.append(mrt_value)

        # Create tuple value containing all result values
        tuple_value = Value(Tuple(result_values))
        node = self.executor.add_op_node(runtime_op, input_nodes, tuple_value)

        # Store the tuple node and also create individual result nodes for getitem access
        tuple_node = node
        for i, result in enumerate(results):
            # Create a getitem node to extract individual result from tuple
            getitem_node = self.executor.add_op_node(
                Op.tuple_getitem,
                [tuple_node, self.executor.add_value_node(Value(i))],
                result_values[i],
            )
            self.env[result] = getitem_node

    def _get_input_nodes(self, op) -> List[Node]:
        """Get input nodes for an operation."""
        input_nodes = []
        for operand in op.operands:
            if operand not in self.env:
                raise RuntimeError(f"Operand for operator {op.name} not ready")
            input_nodes.append(self.env[operand])
        return input_nodes


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
