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
A simple torch.fx backend that converts a GraphModule to a mrt GraphExecutor.
"""
import operator
from functools import reduce
from typing import List, Dict, Any
import sympy
import torch
from torch._ops import OpOverloadPacket
from torch.fx.node import Argument, Node
from torch.fx.graph_module import GraphModule
from torch.utils._sympy.functions import FloorDiv

from mrt import config
from mrt.ir import GraphExecutor, Op, SymbolicVar, SymbolicConst, SymbolicExpr
from mrt.torch.utils import from_torch, to_torch, update_tensor_data, get_collective_info_from_torch, \
    set_device_context


def _init_mrt_config():
    """Initialize the mrt configs."""
    try:
        import torch_npu # pylint: disable=import-outside-toplevel
        config.ascend.op_precision.set_is_allow_matmul_hf32(torch_npu.npu.matmul.allow_hf32)
        # pylint: disable=protected-access
        acl_precision_mode = torch_npu._C._npu_getOption("ACL_PRECISION_MODE")
        config.ascend.op_precision.set_acl_precision_mode(acl_precision_mode.decode())
    except ImportError:
        print("torch_npu is not installed, using default mrt configs.")

# pylint: disable=bad-continuation
def _convert_sympy_expr_to_symbolic_expr(
    expr: sympy.Expr, symbol_map: Dict[sympy.Symbol, SymbolicVar]
) -> SymbolicExpr:
    """Recursively convert a sympy expression to a SymbolicExpr."""
    if expr.is_Symbol:
        if expr not in symbol_map:
            symbol_map[expr] = SymbolicVar(str(expr))
        return symbol_map[expr]

    if expr.is_Integer:
        return SymbolicConst(int(expr))

    if isinstance(expr, FloorDiv):
        base_expr = _convert_sympy_expr_to_symbolic_expr(expr.base, symbol_map)
        divisor_expr = _convert_sympy_expr_to_symbolic_expr(expr.divisor, symbol_map)
        return base_expr // divisor_expr

    # It's an expression with args
    converted_args = [
        _convert_sympy_expr_to_symbolic_expr(arg, symbol_map) for arg in expr.args
    ]

    if expr.is_Add:
        return reduce(operator.add, converted_args)

    if expr.is_Mul:
        return reduce(operator.mul, converted_args)

    raise NotImplementedError(
        f"Unsupported sympy expression: {expr} (type: {type(expr)})"
    )


_GLOBAL_GRAPH_ID = 0

_ARG_MAPPING_HOOKS = {}

def register_arg_mapping_hook(op, hook_func):
    _ARG_MAPPING_HOOKS[op] = hook_func

def  get_arg_mapping_hook(op):
    return _ARG_MAPPING_HOOKS.get(op)

# pylint: disable=unused-argument
def embedding_hook(node, input_nodes, executor):
    """swap the first and second param position."""
    return [input_nodes[1], input_nodes[0]]

def _init_arg_mapping_hooks():
    register_arg_mapping_hook(Op.embedding, embedding_hook)

def _next_unique_graph_id():
    global _GLOBAL_GRAPH_ID
    _GLOBAL_GRAPH_ID += 1
    return _GLOBAL_GRAPH_ID


# pylint: disable=protected-access
# A comprehensive mapping from torch fx ops to our custom ops.
_OP_MAP = {
    # torch functions
    torch.add: Op.add,
    torch.sub: Op.sub,
    torch.mul: Op.mul,
    torch.div: Op.div,
    torch.eq: Op.eq,
    torch.ne: Op.ne,
    torch.lt: Op.lt,
    torch.le: Op.le,
    torch.gt: Op.gt,
    torch.ge: Op.ge,
    torch.matmul: Op.matmul,
    torch.reshape: Op.reshape,
    torch.transpose: Op.transpose,
    torch.cat: Op.concat,
    torch.neg: Op.neg,
    torch.square: Op.square,
    torch.rsqrt: Op.rsqrt,
    torch.relu: Op.relu,
    torch.sigmoid: Op.sigmoid,
    torch.ops._c10d_functional.all_gather_into_tensor: Op.all_gather,
    torch.ops._c10d_functional.all_reduce: Op.all_reduce,
    torch.ops._c10d_functional.reduce_scatter_tensor: Op.reduce_scatter,
    torch.ops._c10d_functional.all_to_all_single: Op.all_to_all,
    torch.ops._c10d_functional.wait_tensor: Op.wait_tensor,
    # torch.nn.functional
    torch.nn.functional.relu: Op.relu,
    torch.nn.functional.sigmoid: Op.sigmoid,
    torch.nn.functional.gelu: Op.gelu,
    torch.nn.functional.silu: Op.silu,
    torch.nn.functional.softmax: Op.softmax,
    torch.nn.functional.layer_norm: Op.norm,
    torch.nn.functional.embedding: Op.embedding,
    # torch.ops.npu functions
    torch.ops.npu.npu_moe_init_routing_v2: Op.moe_init_routing_v3,
    torch.ops.npu.npu_add_rms_norm: Op.add_rms_norm,
    torch.ops.npu.npu_rms_norm: Op.rms_norm,
    # operator functions
    operator.getitem: Op.tuple_getitem,
    operator.add: Op.add,
    operator.sub: Op.sub,
    operator.mul: Op.mul,
    operator.truediv: Op.div,
    operator.eq: Op.eq,
    operator.ne: Op.ne,
    operator.lt: Op.lt,
    operator.le: Op.le,
    operator.gt: Op.gt,
    operator.ge: Op.ge,
    operator.matmul: Op.matmul,
    operator.neg: Op.neg,
    # tensor methods (as strings)
    "size": Op.shape,
    "add": Op.add,
    "sub": Op.sub,
    "mul": Op.mul,
    "div": Op.div,
    "eq": Op.eq,
    "ne": Op.ne,
    "lt": Op.lt,
    "le": Op.le,
    "gt": Op.gt,
    "ge": Op.ge,
    "relu": Op.relu,
    "sigmoid": Op.sigmoid,
    "reshape": Op.reshape,
    "transpose": Op.transpose,
    "neg": Op.neg,
    "square": Op.square,
    "rsqrt": Op.rsqrt,
    "view": Op.reshape,  # view is often used like reshape
    "copy_": Op.copy,
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

        if isinstance(target, torch._ops.OpOverloadPacket):
            node_module = target.__module__
            if not node_module.startswith("torch._ops.aten") and \
               not node_module.startswith("torch._ops.prims"):
                return Op.custom_call
    return None


def _flatten_args(op: Op, node: Node) ->List[Argument]:
    """
    Flatten the arguments of a given FX node into a flat list of Argument objects.

    Args:
        op (Op): The mrt operation enumeration.
        node (Node): The FX node whose arguments should be flattened.

    Returns:
        List[Argument]: A flat list of all Argument objects in the node's arguments, preserving order.
    """
    flat_args = list(node.args)
    # for custom op
    if op == Op.custom_call:
        op_name = node.target.__name__
        flat_args = [op_name] + flat_args
        return flat_args
    kwargs = node.kwargs
    if not kwargs:
        return flat_args
    if not isinstance(node.target, OpOverloadPacket):
        raise RuntimeError(f"Unsupported node target for keyword only args: {node.target}")
    schemas = node.target._schemas
    if len(schemas) != 1:
        raise RuntimeError("Currently, do not support op overload")
    schema = next(iter(schemas.values()))
    for argument in schema.arguments:
        if not argument.kwarg_only:
            continue
        flat_args.append(kwargs[argument.name])
    return flat_args


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
# pylint: disable=unused-argument
def backend(gm: GraphModule, example_inputs: List[torch.Tensor]):
    """
    A torch.fx backend that converts a GraphModule to a da.runtime.GraphExecutor,
    and returns a callable that executes the compiled graph.
    """
    gm.print_readable()
    print("======================fx graph======================")
    print(gm.graph)
    _init_arg_mapping_hooks()
    _init_mrt_config()

    executor = GraphExecutor(f"fx_graph_{_next_unique_graph_id()}")
    env: Dict[Node, Any] = {}
    symbol_map: Dict[sympy.Symbol, SymbolicVar] = {}

    def _create_symbolic_shape_if_needed(example_value, output_value):
        if isinstance(example_value, torch.Tensor) and any(
            isinstance(d, torch.SymInt) for d in example_value.shape
        ):
            symbolic_shape = []
            for dim in example_value.shape:
                if isinstance(dim, torch.SymInt):
                    expr = dim.node.expr
                    symbolic_shape.append(
                        _convert_sympy_expr_to_symbolic_expr(expr, symbol_map)
                    )
                else:
                    symbolic_shape.append(SymbolicConst(int(dim)))
            output_value.to_tensor().symbolic_shape = symbolic_shape

    set_device_context()
    get_collective_info_from_torch(gm)

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            example_value = node.meta.get("example_value", None)
            output_value = from_torch(example_value)
            _create_symbolic_shape_if_needed(example_value, output_value)
            env[node] = executor.add_value_node(output_value)

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

                flat_node_args = _flatten_args(op, node)
                input_nodes = _map_args(flat_node_args, env, executor)
                hook_func = get_arg_mapping_hook(op)
                if hook_func is not None:
                    input_nodes = hook_func(node, input_nodes, executor)
                    print(f"Applied arg mapping hook for {op}, new input nodes:{input_nodes}")
                example_value = node.meta.get("example_value", None)
                output_value = from_torch(example_value)
                _create_symbolic_shape_if_needed(example_value, output_value)

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

        for fx_param_value, mrt_param_node, input_value in zip(
            fx_param_values, mrt_param_nodes, inputs
        ):
            if isinstance(fx_param_value, torch.Tensor):
                update_tensor_data(mrt_param_node.output.to_tensor(), input_value)
            elif isinstance(fx_param_value, torch.SymInt):
                mrt_param_node.output = from_torch(int(input_value))
                expr = fx_param_value.node.expr
                if expr in symbol_map:
                    symbol_map[expr].set_value(int(input_value))
            else:
                mrt_param_node.output = from_torch(input_value)

        result = executor.run()
        return to_torch(result)

    return compiled_callable
