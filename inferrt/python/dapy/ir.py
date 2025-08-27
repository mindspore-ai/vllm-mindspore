from typing import List
import numpy as np
from _dairpy import GraphExecutor as _GraphExecutor, Node, Op


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

    def add_op(self, op: Op, inputs: List[Node]) -> Node:
        """Add an operation node to the graph."""
        return self._executor.add_op(op, inputs)

    def make_tuple(self, inputs: List[Node]) -> Node:
        """Add a make_tuple operation to the graph."""
        node = self._executor.add_op(Op.make_tuple, inputs)
        return node

    def add_const(self, node: np.ndarray) -> Node:
        """Add a constant node to the graph from a numpy array."""
        return self._executor.add_const(node)

    def set_return(self) -> Node:
        """Add a return node to the graph. The last added node will be the return value."""
        self._return_node = self._executor.add_return()
        return self._return_node

    def run(self, is_dynamic: bool = True) -> List[np.ndarray]:
        """Run the built graph and return the output as a numpy array List."""
        self._executor.run_graph(is_dynamic)
        if self._return_node is None:
            raise RuntimeError("Return node not set. Call set_return() before running.")
        result = self._return_node.list()
        self._executor.free_graph_outputs()
        return result

    def build(self):
        """Optimize the graph and build kernels."""
        # self._executor.opt_graph()
        self._executor.build_kernels()
        self._executor.record_tensor_ref_count()

    def dump_graph(self):
        """Dump the graph definition."""
        self._executor.dump_graph()


# Re-export for convenience
__all__ = ["GraphExecutor", "Node", "Op"]
