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
A Python wrapper for the mrt C++ GraphExecutor to build and run computation graphs.
"""
from typing import List
from mrt._mrt_ir import (
    GraphExecutor as _GraphExecutor,
    Node,
    Op,
    Tensor,
    Value,
    Tuple,
    SymbolicVar,
    SymbolicConst,
    SymbolicExpr,
    DataType,
    Device,
    DeviceType,
)


class GraphExecutor:
    """
    A Python wrapper for the C++ GraphExecutor to build and run computation graphs.

    This class can be used as a context manager:

    with GraphExecutor("my_graph") as executor:
        param = executor.add_parameter(node)
        ...
        executor.run()
    """

    def __init__(self, name: str = "default_graph"):
        self._executor = _GraphExecutor()
        self._name = name
        self._return_node = None

    def __enter__(self):
        self._executor.begin_graph(self._name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._executor.end_graph()

    def add_parameter(self, node: Node) -> Node:
        """Add a const node as parameter of the graph."""
        self._executor.add_parameter(node)

    def add_op_node(self, op: Op, inputs: List[Node], output: Value) -> Node:
        """Add an operation node to the graph."""
        if op == Op.tuple_getitem:
            output = self._tuple_getitem(inputs)
        return self._executor.add_op_node(op, inputs, output)

    def _tuple_getitem(self, inputs: List[Node]) -> Value:
        """Get a value from a tuple, only used for Op.tuple_getitem."""
        if len(inputs) != 2:
            raise ValueError("Tuple getitem requires exactly 2 inputs")
        return inputs[0].output.to_tuple()[inputs[1].output.to_int()]

    def make_tuple(self, inputs: List[Node]) -> Node:
        """Add a make_tuple operation to the graph."""
        output = Value(Tuple([input.output for input in inputs]))
        node = self._executor.add_op_node(Op.make_tuple, inputs, output)
        return node

    def add_value_node(self, value: Value) -> Node:
        """Add a constant node to the graph from a python object (e.g. torch.Tensor, scalar)."""
        return self._executor.add_value_node(value)

    def add_return_node(self, node: Node):
        """Add a return node to the graph."""
        self._return_node = node
        self._executor.add_return_node(node)

    def run(self, is_dynamic: bool = True) -> Value:
        """Run the built graph and return the output."""
        self._executor.run_graph(is_dynamic)
        if self._return_node is None:
            raise RuntimeError("Return node not set. Call add_return_node() before running.")
        return self._return_node.output

    def build(self):
        """Optimize the graph and build kernels."""
        # self._executor.opt_graph()
        self._executor.build_executor()

    def dump_graph(self):
        """Dump the graph definition."""
        self._executor.dump_graph()


# Re-export for convenience
__all__ = [
    "GraphExecutor",
    "Node",
    "Op",
    "Tensor",
    "Value",
    "Tuple",
    "SymbolicVar",
    "SymbolicConst",
    "SymbolicExpr",
    "DataType",
    "Device",
    "DeviceType",
]
