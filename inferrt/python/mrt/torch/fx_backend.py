"""
A simple torch.fx backend that converts a GraphModule to a mrt GraphExecutor.
"""

import operator
from typing import List, Dict, Any
import sympy
import torch
from torch.fx.node import Node
from torch.fx.graph_module import GraphModule

from mrt.ir import GraphExecutor, Op
from mrt.torch.utils import from_torch, to_torch, update_tensor_data


_GLOBAL_GRAPH_ID = 0


def _next_unique_graph_id():
    global _GLOBAL_GRAPH_ID
    _GLOBAL_GRAPH_ID += 1
    return _GLOBAL_GRAPH_ID


# A comprehensive mapping from torch fx ops to our custom ops.
_OP_MAP = {
    # torch functions
    torch.add: Op.add,
    torch.sub: Op.sub,
    torch.mul: Op.mul,
    torch.div: Op.div,
    torch.matmul: Op.matmul,
    torch.reshape: Op.reshape,
    torch.transpose: Op.transpose,
    torch.cat: Op.concat,
    torch.neg: Op.neg,
    torch.square: Op.square,
    torch.rsqrt: Op.rsqrt,
    torch.relu: Op.relu,
    torch.sigmoid: Op.sigmoid,
    # torch.nn.functional
    torch.nn.functional.relu: Op.relu,
    torch.nn.functional.sigmoid: Op.sigmoid,
    torch.nn.functional.gelu: Op.gelu,
    torch.nn.functional.silu: Op.silu,
    torch.nn.functional.softmax: Op.softmax,
    torch.nn.functional.layer_norm: Op.norm,
    # operator functions
    operator.add: Op.add,
    operator.sub: Op.sub,
    operator.mul: Op.mul,
    operator.truediv: Op.div,
    operator.matmul: Op.matmul,
    operator.neg: Op.neg,
    # tensor methods (as strings)
    "size": Op.shape,
    "add": Op.add,
    "sub": Op.sub,
    "mul": Op.mul,
    "div": Op.div,
    "relu": Op.relu,
    "sigmoid": Op.sigmoid,
    "reshape": Op.reshape,
    "transpose": Op.transpose,
    "neg": Op.neg,
    "square": Op.square,
    "rsqrt": Op.rsqrt,
    "view": Op.reshape,  # view is often used like reshape
}


def _get_op(target):
    """Get the corresponding Op enum for a given target."""
    if isinstance(target, str):
        return _OP_MAP.get(target)
    if callable(target):
        op = _OP_MAP.get(target)
        if op is not None:
            return op
        # For torch ops that are not in _OP_MAP, try to get their name
        # and look up in the Op enum. This is more generic.
        if hasattr(target, "__name__"):
            op_name = target.__name__
            if hasattr(Op, op_name):
                return getattr(Op, op_name)
    return None


def _map_args(args, env, executor: GraphExecutor) -> List[Node]:
    """
    Map torch.fx node arguments to GraphExecutor nodes.
    This function handles nested structures like lists and tuples.
    """

    def _map_arg(arg: Any) -> Node:
        if isinstance(arg, Node):
            return env[arg]

        if isinstance(arg, (list, tuple)):
            nodes = [_map_arg(item) for item in arg]
            return executor.make_tuple(nodes)

        return executor.add_value_node(from_torch(arg))

    return [_map_arg(arg) for arg in args]


# pylint: disable=bad-continuation
def backend(gm: GraphModule, example_inputs: List[torch.Tensor]):
    """
    A torch.fx backend that converts a GraphModule to a da.runtime.GraphExecutor,
    and returns a callable that executes the compiled graph.
    """
    gm.print_readable()

    executor = GraphExecutor(f"fx_graph_{_next_unique_graph_id()}")
    env: Dict[Node, Any] = {}

    input_iterator = iter(example_inputs)

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            env[node] = executor.add_value_node(from_torch(next(input_iterator)))

    with executor:
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                executor.add_parameter(env[node])

            elif node.op == "get_attr":
                target = node.target
                assert isinstance(target, str)

                attr_val = gm
                for part in target.split("."):
                    attr_val = getattr(attr_val, part)

                env[node] = executor.add_value_node(from_torch(attr_val))

            elif node.op in ("call_function", "call_method"):
                op = _get_op(node.target)
                if op is None:
                    raise NotImplementedError(f"Unsupported op: {node.target}")

                input_nodes = _map_args(node.args, env, executor)
                output_value = from_torch(node.meta.get("example_value", None))

                env[node] = executor.add_op_node(op, input_nodes, output_value)

            elif node.op == "call_module":
                raise NotImplementedError(
                    "call_module is not supported in this simple backend."
                )

            elif node.op == "output":
                input_nodes = _map_args(node.args, env, executor)
                env[node] = input_nodes[0]
                executor.set_return()

            else:
                raise NotImplementedError(f"Unsupported node op: {node.op}")

    print("Building Graph:")
    executor.dump_graph()
    executor.build()

    fx_param_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
    fx_param_values = [n.meta["example_value"] for n in fx_param_nodes]
    mrt_param_nodes = [env[n] for n in fx_param_nodes]

    def compiled_callable(*inputs: torch.Tensor):
        if len(inputs) != len(mrt_param_nodes):
            raise ValueError(
                f"Expected {len(mrt_param_nodes)} inputs, but got {len(inputs)}"
            )

        sym_bindings = {}
        for fx_param_value, mrt_param_node, input_value in zip(
            fx_param_values, mrt_param_nodes, inputs
        ):
            if isinstance(fx_param_value, torch.Tensor):
                update_tensor_data(mrt_param_node.output.to_tensor(), input_value)
            elif isinstance(fx_param_value, torch.SymInt):
                sym_bindings[fx_param_value.node.expr] = input_value
            else:
                mrt_param_node.output = from_torch(input_value)

        def sym_eval(value):
            if isinstance(value, torch.SymInt):
                value = sympy.expand(value.node.expr.xreplace(sym_bindings))
                if value.is_number:
                    value = value.evalf()
            return value

        for fx_node, mrt_node in env.items():
            if (
                fx_node.op in ("call_function", "call_method")
                and mrt_node.output.is_tensor()
            ):
                output_shape = fx_node.meta.get("example_value", None).shape
                mrt_node.output.to_tensor().shape = [
                    sym_eval(dim) for dim in output_shape
                ]

        print("Running Graph:")
        executor.dump_graph()
        result = executor.run()
        return to_torch(result)

    return compiled_callable
