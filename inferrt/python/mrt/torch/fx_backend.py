import torch
import operator
from typing import List, Dict, Any
from torch.fx.node import Node, map_arg
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
    if isinstance(target, str):
        return _OP_MAP.get(target)
    if callable(target):
        op = _OP_MAP.get(target)
        if op is not None:
            return op
        # For torch ops that are not in _OP_MAP, try to get their name
        # and look up in the Op enum. This is more generic.
        if hasattr(target, "_overloadpacket"):
            op_name = target._overloadpacket.__name__
            if hasattr(Op, op_name):
                return getattr(Op, op_name)
        if hasattr(target, "__name__"):
            op_name = target.__name__
            if hasattr(Op, op_name):
                return getattr(Op, op_name)
    return None


def _map_args(node_args, env):
    return map_arg(node_args, lambda arg: env[arg] if isinstance(arg, Node) else arg)


def backend(gm: GraphModule, example_inputs: List[torch.Tensor]):
    """
    A torch.fx backend that converts a GraphModule to a da.runtime.GraphExecutor,
    and returns a callable that executes the compiled graph.
    """
    executor = GraphExecutor(f"fx_graph_{_next_unique_graph_id()}")
    env: Dict[Node, Any] = {}

    input_iterator = iter(example_inputs)

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            input = next(input_iterator)
            if isinstance(input, torch.Tensor):
                input = from_torch(input)
            env[node] = executor.add_value_node(input)

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

                if isinstance(attr_val, torch.Tensor):
                    attr_val = from_torch(attr_val)

                env[node] = executor.add_value_node(attr_val)

            elif node.op in ("call_function", "call_method"):
                op = _get_op(node.target)
                if op is None:
                    raise NotImplementedError(f"Unsupported op: {node.target}")

                args = _map_args(node.args, env)

                env[node] = executor.add_op_node(op, list(args))

            elif node.op == "call_module":
                raise NotImplementedError(
                    "call_module is not supported in this simple backend."
                )

            elif node.op == "output":
                output_nodes = node.args[0]
                outputs = list(_map_args(output_nodes, env))
                env[node] = executor.make_tuple(outputs)
                executor.set_return()

            else:
                raise NotImplementedError(f"Unsupported node op: {node.op}")

    executor.dump_graph()
    executor.build()

    placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
    param_nodes = [env[n] for n in placeholder_nodes]

    def compiled_callable(*new_inputs: torch.Tensor):
        if len(new_inputs) != len(param_nodes):
            raise ValueError(
                f"Expected {len(param_nodes)} inputs, but got {len(new_inputs)}"
            )

        for i, p_node in enumerate(param_nodes):
            update_tensor_data(p_node.output.to_tensor(), new_inputs[i])

        result = executor.run()

        return tuple(to_torch(r) for r in result)

    return compiled_callable
