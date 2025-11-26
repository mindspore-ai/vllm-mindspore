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

"""
Symbolic shape management for handling dynamic tensor dimensions.
"""
import operator
from functools import reduce
from typing import List, Dict, Any, Optional

import torch
import sympy

from torch.utils._sympy.functions import (
    FloorDiv,
    IntTrueDiv,
    CeilDiv,
    FloorToInt,
    CeilToInt,
)
from mrt.ir import SymbolicVar, SymbolicConst, SymbolicExpr, Value


class SymbolicShapeManager:
    """
    Manages symbolic shape creation and conversion for tensors with dynamic dimensions.

    This class handles the conversion between sympy expressions and MRT symbolic expressions,
    and recursively processes nested tensor structures (lists, tuples) to create symbolic shapes.
    """

    _SYMPY_EXPR_TO_SYMBOLIC_OP = {
        sympy.Add: operator.add,
        sympy.Mul: operator.mul,
        FloorDiv: operator.floordiv,
        CeilDiv: lambda a, b: a.__ceildiv__(b),
    }

    def __init__(self):
        self._symbol_map: Dict[sympy.Symbol, SymbolicVar] = {}

    def convert_sympy_expr_to_symbolic_expr(self, expr: sympy.Expr) -> SymbolicExpr:
        """
        Recursively convert a sympy expression to a SymbolicExpr.

        Args:
            expr: The sympy expression to convert

        Returns:
            The corresponding SymbolicExpr

        Raises:
            NotImplementedError: If the expression type is not supported
        """
        if expr.is_Symbol:
            if expr not in self._symbol_map:
                self._symbol_map[expr] = SymbolicVar(str(expr))
            return self._symbol_map[expr]

        if expr.is_Integer:
            return SymbolicConst(int(expr))

        # Eliminate ops with float outputs
        if isinstance(expr, FloorToInt) and isinstance(expr.args[0], IntTrueDiv):
            return self.convert_sympy_expr_to_symbolic_expr(
                FloorDiv(*expr.args[0].args)
            )

        if isinstance(expr, CeilToInt) and isinstance(expr.args[0], IntTrueDiv):
            return self.convert_sympy_expr_to_symbolic_expr(CeilDiv(*expr.args[0].args))

        # Convert basic symbolic ops
        op = self._SYMPY_EXPR_TO_SYMBOLIC_OP.get(type(expr))
        if op:
            converted_args = [
                self.convert_sympy_expr_to_symbolic_expr(arg) for arg in expr.args
            ]
            if isinstance(expr, (sympy.Add, sympy.Mul)):
                return reduce(op, converted_args)
            return op(*converted_args)

        raise NotImplementedError(
            f"Unsupported sympy expression: {expr} (type: {type(expr)})"
        )

    def create_symbolic_shape_for_tensor(self, tensor) -> Optional[List[SymbolicExpr]]:
        """
        Create symbolic shape for a tensor if it contains symbolic dimensions.

        Args:
            tensor: The torch tensor that may contain symbolic dimensions

        Returns:
            List of symbolic expressions for the tensor shape, or None if no symbolic dimensions
        """
        if not isinstance(tensor, torch.Tensor):
            return None

        if not any(isinstance(d, torch.SymInt) for d in tensor.shape):
            return None

        symbolic_shape = []
        for dim in tensor.shape:
            if isinstance(dim, torch.SymInt):
                expr = dim.node.expr
                symbolic_shape.append(self.convert_sympy_expr_to_symbolic_expr(expr))
            else:
                symbolic_shape.append(SymbolicConst(int(dim)))

        return symbolic_shape

    def bind_symbolic_shape(self, mrt_value: Value, torch_value: Any) -> None:
        """
        Recursively create and apply symbolic shape for tensors in nested structures.

        Args:
            mrt_value: The corresponding MRT value (can be Tensor, Tuple, or nested structures)
            torch_value: The value from torch (can be Tensor, list, tuple, or nested structures)
        """
        # Handle tensor case
        if isinstance(torch_value, torch.Tensor) and mrt_value.is_tensor():
            symbolic_shape = self.create_symbolic_shape_for_tensor(torch_value)
            if symbolic_shape is not None:
                mrt_value.to_tensor().symbolic_shape = symbolic_shape

        # Handle nested lists/tuples recursively
        elif isinstance(torch_value, (list, tuple)) and mrt_value.is_tuple():
            tuple_value = mrt_value.to_tuple()
            if len(tuple_value) != len(torch_value):
                return
            for i, torch_item in enumerate(torch_value):
                self.bind_symbolic_shape(tuple_value[i], torch_item)

    def set_symbolic_var(self, symbol: sympy.Symbol, value: int) -> bool:
        """
        Set the value for a symbolic variable.

        Args:
            symbol: The sympy symbol
            value: The integer value to set

        Returns:
            True if the symbol was found and value was set, False otherwise
        """
        symbolic_var = self._symbol_map.get(symbol)
        if symbolic_var is not None:
            symbolic_var.set_value(value)
            return True
        return False
