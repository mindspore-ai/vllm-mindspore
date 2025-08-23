from typing import List
import numpy as np
from _dairpy import GraphExecutor as _GraphExecutor, DATensor, Op


class GraphExecutor:
    """
    A Python wrapper for the C++ GraphExecutor to build and run computation graphs.

    This class can be used as a context manager:

    with GraphExecutor("my_graph") as executor:
        param = executor.add_parameter(tensor)
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

    def add_parameter(self, tensor: DATensor) -> DATensor:
        """Add a const tensor as parameter of the graph."""
        self._executor.add_parameter(tensor)

    def add_op(self, op: Op, inputs: List[DATensor]) -> DATensor:
        """Add an operation tensor to the graph."""
        return self._executor.add_op(op, inputs)

    def make_tuple(self, inputs: List[DATensor]) -> DATensor:
        """Add a make_tuple operation to the graph."""
        tensor = self._executor.add_op(Op.make_tuple, inputs)
        self._executor.cast_to_tensor_list(tensor, len(inputs))
        return tensor

    def add_const(self, tensor: np.ndarray) -> DATensor:
        """Add a constant tensor to the graph from a numpy array."""
        return self._executor.add_const(tensor)

    def set_return(self) -> DATensor:
        """Add a return node to the graph. The last added tensor will be the return value."""
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
__all__ = ["GraphExecutor", "DATensor", "Op"]
