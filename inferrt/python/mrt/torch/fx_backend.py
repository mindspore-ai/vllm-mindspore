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
from typing import List, Dict, Any, Optional
import torch
from torch._ops import OpOverload, OpOverloadPacket
from torch.fx.node import Argument, Node
from torch.fx.graph_module import GraphModule

from mrt import config
from mrt.ir import GraphExecutor, Op
from mrt.torch.symbolic_shape import SymbolicShapeManager
from mrt.torch.utils import (
    from_torch,
    to_torch,
    get_collective_info_from_torch,
    set_device_context,
    update_runtime_inputs,
)

try:
    import torch_npu  # pylint: disable=import-outside-toplevel,unused-import

    TORCH_NPU_INSTALLED = True
except ImportError:
    TORCH_NPU_INSTALLED = False


def _init_mrt_config():
    """Initialize the mrt configs."""
    if TORCH_NPU_INSTALLED:
        config.ascend.op_precision.set_is_allow_matmul_hf32(
            torch_npu.npu.matmul.allow_hf32
        )
        # pylint: disable=protected-access
        acl_precision_mode = torch_npu._C._npu_getOption("ACL_PRECISION_MODE")
        config.ascend.op_precision.set_acl_precision_mode(acl_precision_mode.decode())
    else:
        print("torch_npu is not installed, using default mrt configs.")


_GLOBAL_GRAPH_ID = 0

_ARG_MAPPING_HOOKS = {}

_OPS_MAPPING_HOOKS = {}

# Registry for linalg ops: maps op_name -> mlir_text
# TODO(lmy) this temporary interface will be removed when mrt backend is ready.
_LINALG_OP_REGISTRY = {}


def register_linalg_op(op_name: str, mlir_text: str):
    """Register a linalg op with its MLIR text.
    this temporary interface will be removed when mrt backend is ready.

    Args:
        op_name: The operator name (e.g., "linalg_add")
        mlir_text: The Linalg MLIR text with hacc annotations
    """
    _LINALG_OP_REGISTRY[op_name] = mlir_text


def get_linalg_mlir(op_name: str) -> str:
    """Get MLIR text for a registered linalg op.
    this temporary interface will be removed when mrt backend is ready.
    """
    return _LINALG_OP_REGISTRY.get(op_name)


def register_arg_mapping_hook(op, hook_func):
    _ARG_MAPPING_HOOKS[op] = hook_func


def get_arg_mapping_hook(op):
    return _ARG_MAPPING_HOOKS.get(op)

def register_ops_mapping_hook(op, hook_func):
    _OPS_MAPPING_HOOKS[op] = hook_func

def get_ops_mapping_hook(op):
    return _OPS_MAPPING_HOOKS.get(op)


# pylint: disable=unused-argument
def embedding_hook(node, input_nodes, executor):
    """swap the first and second param position."""
    return [input_nodes[1], input_nodes[0]]

# pylint: disable=unused-argument
def apply_rotary_pos_emb_hook(node, input_nodes, executor):
    """add layout parameter."""
    rope_layout_bsnd = 1
    return [input_nodes[0], input_nodes[1], input_nodes[2], input_nodes[3], rope_layout_bsnd]

# pylint: disable=unused-argument
def floor_div_hook(node, input_nodes, executor):
    """add div mode parameter."""
    div_mode = 2
    return [input_nodes[0], input_nodes[1], div_mode]

# pylint: disable=unused-argument
def long_hook(node, input_nodes, executor):
    """add long."""
    return [input_nodes[0],  torch.int64]

# pylint: disable=unused-argument
def permute_hook(node, input_nodes, executor):
    """transpose dims"""
    dim_inx = list(range(0, len(input_nodes[0].meta["example_value"].shape), 1))
    dim_inx[input_nodes[1]] = input_nodes[2]
    dim_inx[input_nodes[2]] = input_nodes[1]

    return [input_nodes[0],  dim_inx]

def _init_arg_mapping_hooks():
    register_arg_mapping_hook(Op.permute, permute_hook)
    register_arg_mapping_hook(Op.embedding, embedding_hook)
    register_arg_mapping_hook(Op.apply_rotary_pos_emb, apply_rotary_pos_emb_hook)
    register_arg_mapping_hook(operator.floordiv, floor_div_hook)
    register_arg_mapping_hook("long", long_hook)

# pylint: disable=unused-argument
def split_ops_hook(op, node, input_nodes, executor):
    if isinstance(input_nodes[1], (int, torch.SymInt)):
        return Op.split_tensor
    if hasattr(input_nodes[1], "meta") and input_nodes[1].meta is not None:
        if isinstance(input_nodes[1].meta["example_value"], (int, torch.SymInt)):
            return Op.split_tensor
    return op

def _init_ops_mapping_hooks():
    register_ops_mapping_hook(Op.split_with_size, split_ops_hook)

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
    torch.transpose: Op.permute,
    torch.unsqueeze: Op.unsqueeze,
    torch.split: Op.split_with_size,
    torch.flatten: Op.flatten,
    torch.cat: Op.cat,
    torch.neg: Op.neg,
    torch.square: Op.square,
    torch.rsqrt: Op.rsqrt,
    torch.relu: Op.relu,
    torch.sigmoid: Op.sigmoid,
    torch.empty: Op.empty,
    torch.zeros: Op.zeros,
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
    operator.and_: Op.bitwise_and_tensor,
    operator.invert: Op.bitwise_not,
    operator.mod: Op.remainder_tensor_tensor,
    operator.floordiv: Op.div_mod,
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
    "to": Op.cast,
    "sigmoid": Op.sigmoid,
    "reshape": Op.reshape,
    "cat": Op.cat,
    "transpose": Op.permute,
    "unsqueeze": Op.unsqueeze,
    "neg": Op.neg,
    "square": Op.square,
    "rsqrt": Op.rsqrt,
    "view": Op.reshape,  # view is often used like reshape
    "copy_": Op.copy,
    "long": Op.cast,
    "split": Op.split_with_size,
    "flatten": Op.flatten,
}

if TORCH_NPU_INSTALLED:
    _NPU_OP_MAP = {
        # torch.ops.npu functions
        torch.ops.npu.npu_moe_init_routing_v2: Op.moe_init_routing_v3,
        torch.ops.npu.npu_add_rms_norm: Op.add_rms_norm,
        torch.ops.npu.npu_rms_norm: Op.rms_norm,
        torch.ops.npu.npu_scatter_nd_update: Op.scatter_nd_update,
        torch.ops.npu.npu_moe_token_unpermute: Op.moe_token_unpermute,
        torch.ops.npu.npu_swiglu: Op.swiglu,
        torch.ops.npu.npu_moe_gating_top_k_softmax: Op.moe_gating_top_k_softmax,
        torch.ops.npu.npu_apply_rotary_pos_emb: Op.apply_rotary_pos_emb,
        torch.ops.npu.npu_grouped_matmul: Op.grouped_matmul,
    }
    _OP_MAP.update(_NPU_OP_MAP)


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
            if not node_module.startswith(
                "torch._ops.aten"
            ) and not node_module.startswith("torch._ops.prims"):
                # mrt_linalg namespace -> linalg_call, mrt namespace -> custom_call
                # TODO(lmy) this temporary interface will be removed when mrt backend is ready.
                if node_module.startswith("torch._ops.mrt_linalg"):
                    return Op.linalg_call
                return Op.custom_call
    return None

def _argument_to_real_value(value_type, value, arg_len):
    """
    Convert a torch fx value to its real value.

    Args:
        value_type (torch.dtype): The type of the value.
        value (Any): The value of the argument.

    Returns:
        Any: The real value of the argument.
    """
    if isinstance(value_type, torch.OptionalType):
        return _argument_to_real_value(value_type.getElementType(), value, arg_len)
    if isinstance(value_type, torch.ListType):
        if isinstance(value, list):
            return value
        if value is None:
            return value
        if not arg_len:
            return [value]
        return [value for _ in range(arg_len)]
    return value


def _create_args(schema: torch.FunctionSchema, node: Node) -> List[Argument]:
    """
    Create a list of Argument objects from a torch fx node.

    Args:
        schema (torch.FunctionSchema): The schema of the node.
        node (torch.fx.Node): The FX node whose arguments should be created.

    Returns:
        List[Argument]: A list of Argument objects in the node's arguments, preserving order.
        Bool: Whether the arguments are valid.
    """
    flat_args = []
    args = node.args
    kwargs = node.kwargs
    arg_idx = 0
    if len(args) + len(kwargs) > len(schema.arguments):
        return flat_args, False

    for arg in args:
        if schema.arguments[arg_idx].kwarg_only:
            return flat_args, False
        real_arg = _argument_to_real_value(schema.arguments[arg_idx].real_type, arg, schema.arguments[arg_idx].N)
        flat_args.append(real_arg)
        arg_idx += 1

    consumed_kwargs = 0
    for argument in schema.arguments[arg_idx:]:
        if argument.name in kwargs:
            real_arg = _argument_to_real_value(argument.real_type, kwargs[argument.name], argument.N)
            flat_args.append(real_arg)
            consumed_kwargs += 1
        elif hasattr(argument, "default_value"):
            flat_args.append(argument.default_value)
        else:
            return flat_args, False

    if consumed_kwargs != len(kwargs):
        return flat_args, False
    return flat_args, True


def _get_op_schemas(target) -> Optional[List[torch._C.FunctionSchema]]:
    """
    Retrieve torch schema(s) for a given op target. Returns None if unavailable.
    """
    if isinstance(target, str):
        for ns in iter(torch.ops):
            ops_ns = getattr(torch.ops, ns)
            if hasattr(ops_ns, target):
                op_target = getattr(ops_ns, target)
                return [getattr(op_target, overload)._schema for overload in op_target.overloads()]
        return None

    if isinstance(target, OpOverload):
        return [target._schema]

    if isinstance(target, OpOverloadPacket):
        return [getattr(target, overload)._schema for overload in target.overloads()]

    aten_fn = torch.jit._builtins._find_builtin(target)
    if aten_fn is not None:
        return torch._C._jit_get_schemas_for_operator(aten_fn)

    return None


def _flatten_args(op: Op, node: Node) -> List[Argument]:
    """
    Flatten the arguments of a given FX node into a flat list of Argument objects.

    Args:
        op (Op): The mrt operation enumeration.
        node (Node): The FX node whose arguments should be flattened.

    Returns:
        List[Argument]: A flat list of all Argument objects in the node's arguments, preserving order.
    """
    flat_args = []
    schemas = _get_op_schemas(node.target)
    if not schemas:
        return list(node.args) + list(node.kwargs.values())
    found = False
    for schema in schemas:
        flat_args, found = _create_args(schema, node)
        if found:
            break
    if not found:
        err_msg = f"Failed to find a valid schema for {node.target} with arguments {node.args} and kwargs {node.kwargs}"
        raise ValueError(err_msg)
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


def _handle_param_node(node, executor, symbolic_shape_manager, env):
    """Handle param node processing."""
    example_value = node.meta.get("example_value", None)
    output_value = from_torch(example_value)
    symbolic_shape_manager.bind_symbolic_shape(output_value, example_value)
    env[node] = executor.add_value_node(output_value)


def _handle_param_nodes(param_nodes, executor, symbolic_shape_manager, env):
    """Handle param nodes processing."""
    non_symbol_param_nodes = []
    # handle sym int param nodes first to register symbols for later reference
    for node in param_nodes:
        if isinstance(node.meta.get("example_value"), torch.SymInt):
            _handle_param_node(node, executor, symbolic_shape_manager, env)
        else:
            non_symbol_param_nodes.append(node)
    # handle non sym int param nodes
    for node in non_symbol_param_nodes:
        _handle_param_node(node, executor, symbolic_shape_manager, env)


def _handle_get_attr_node(node, gm, executor, env):
    """Handle get_attr node processing."""
    target = node.target
    assert isinstance(target, str)

    attr_val = gm
    for part in target.split("."):
        attr_val = getattr(attr_val, part)

    env[node] = executor.add_value_node(from_torch(attr_val))


def _prepare_call_args(op, node, executor, env):
    """Prepare arguments for call_function/call_method nodes."""
    flat_node_args = _flatten_args(op, node)

    if op == Op.custom_call:
        op_name = node.target.__name__
        flat_node_args = [op_name] + flat_node_args
    elif op == Op.linalg_call:
        op_name = node.target.__name__
        mlir_text = get_linalg_mlir(op_name)
        if mlir_text is None:
            raise RuntimeError(
                f"MLIR not registered for linalg op '{op_name}'. "
                f"Use register_linalg_op('{op_name}', mlir_text) first."
            )
        flat_node_args = [mlir_text] + flat_node_args

    hook_func = get_arg_mapping_hook(op) or get_arg_mapping_hook(node.target)
    if hook_func is not None:
        flat_node_args = hook_func(node, flat_node_args, executor)
        print(f"Applied arg mapping hook for {op}, new input nodes:{flat_node_args}")

    return _map_args(flat_node_args, env, executor)


def _handle_call_node(node, executor, symbolic_shape_manager, env):
    """Handle call_function/call_method node processing."""
    op = _get_op(node.target)
    if op is None:
        raise NotImplementedError(f"Unsupported op: {node.target}")

    ops_hook = get_ops_mapping_hook(op)
    if ops_hook is not None:
        flat_node_args = _flatten_args(op, node)
        op = ops_hook(op, node, flat_node_args, executor)

    input_nodes = _prepare_call_args(op, node, executor, env)
    example_value = node.meta.get("example_value", None)
    output_value = from_torch(example_value)
    symbolic_shape_manager.bind_symbolic_shape(output_value, example_value)

    env[node] = executor.add_op_node(op, input_nodes, output_value)


def _handle_output_node(node, executor, env):
    """Handle output node processing."""
    input_nodes = _map_args(node.args, env, executor)
    env[node] = input_nodes[0]
    executor.set_return()


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
    _init_ops_mapping_hooks()
    _init_mrt_config()

    executor = GraphExecutor(f"fx_graph_{_next_unique_graph_id()}")
    symbolic_shape_manager = SymbolicShapeManager()
    env: Dict[Node, Any] = {}

    get_collective_info_from_torch(gm)
    set_device_context()

    fx_param_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
    _handle_param_nodes(fx_param_nodes, executor, symbolic_shape_manager, env)

    with executor:
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                executor.add_parameter(env[node])
            elif node.op == "get_attr":
                _handle_get_attr_node(node, gm, executor, env)
            elif node.op in ("call_function", "call_method"):
                _handle_call_node(node, executor, symbolic_shape_manager, env)
            elif node.op == "call_module":
                raise NotImplementedError(
                    "call_module is not supported in this simple backend."
                )
            elif node.op == "output":
                _handle_output_node(node, executor, env)
            else:
                raise NotImplementedError(f"Unsupported node op: {node.op}")

    print("Building Graph:")
    executor.dump_graph()
    executor.build()

    mrt_param_nodes = [env[n] for n in fx_param_nodes]

    def compiled_callable(*inputs: torch.Tensor):
        update_runtime_inputs(mrt_param_nodes, inputs)
        result = executor.run()
        return to_torch(result)

    return compiled_callable
