from typing import List
from _mrt_ir import GraphExecutor as _GraphExecutor, Node, Op, Tensor, Value, Tuple


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
        return self._executor.add_op_node(op, inputs, output)

    def make_tuple(self, inputs: List[Node]) -> Node:
        """Add a make_tuple operation to the graph."""
        output = Value(Tuple([input.output for input in inputs]))
        node = self._executor.add_op_node(Op.make_tuple, inputs, output)
        return node

    def add_value_node(self, value: Value) -> Node:
        """Add a constant node to the graph from a python object (e.g. torch.Tensor, scalar)."""
        return self._executor.add_value_node(value)

    def set_return(self) -> Node:
        """Add a return node to the graph. The last added node will be the return value."""
        self._return_node = self._executor.add_return()
        return self._return_node

    def run(self, is_dynamic: bool = True) -> Value:
        """Run the built graph and return the output."""
        self._executor.run_graph(is_dynamic)
        if self._return_node is None:
            raise RuntimeError("Return node not set. Call set_return() before running.")
        return self._return_node.output

    def build(self):
        """Optimize the graph and build kernels."""
        # self._executor.opt_graph()
        self._executor.build_executor()

    def dump_graph(self):
        """Dump the graph definition."""
        self._executor.dump_graph()


# Re-export for convenience
__all__ = ["GraphExecutor", "Node", "Op", "Tensor", "Value", "Tuple"]