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
A simple torch.fx backend that converts a GraphModule to a ms_inferrt GraphExecutor.
"""
import os
import operator
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
import torch
from torch._ops import OpOverload, OpOverloadPacket
from torch.fx.node import Argument, Node
from torch.fx.graph_module import GraphModule
from torch.fx.immutable_collections import immutable_list

from ms_inferrt import _ms_inferrt_ir
from ms_inferrt import config
from ms_inferrt.ir import GraphExecutor, Op
from ms_inferrt.torch.symbolic_shape import SymbolicShapeManager
from ms_inferrt.torch.utils import (
    from_torch,
    to_torch,
    get_collective_info_from_torch,
    set_device_context,
    update_runtime_inputs,
    is_op_registered_by_custom_or_torch,
    get_tensor_arg_dtype,
)
from ms_inferrt.torch.getitem_impl import getitem_process
from ms_inferrt.torch.setitem_impl import setitem_process
from ms_inferrt.torch.decompose_impl import _decompose_ops_with_fake_mode
from ms_inferrt.torch.copy_elimination import eliminate_redundant_copy_
from ms_inferrt.torch.full_decomposition import decompose_full_

try:
    import torch_npu  # pylint: disable=import-outside-toplevel,unused-import

    TORCH_NPU_INSTALLED = True
except ImportError:
    TORCH_NPU_INSTALLED = False


def _debug_print(*args, **kwargs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    pid = os.getpid()
    kwargs["flush"] = True
    print(f"[{timestamp}] [PID:{pid}]", *args, **kwargs)


def _init_ms_inferrt_config():
    """Initialize the ms_inferrt configs."""
    if TORCH_NPU_INSTALLED:
        config.ascend.op_precision.set_is_allow_matmul_hf32(
            torch_npu.npu.matmul.allow_hf32
        )
        # pylint: disable=protected-access
        acl_precision_mode = torch_npu._C._npu_getOption("ACL_PRECISION_MODE")
        config.ascend.op_precision.set_acl_precision_mode(acl_precision_mode.decode())
    else:
        print("torch_npu is not installed, using default ms_inferrt configs.")


_GLOBAL_GRAPH_ID = 0

_ARG_MAPPING_HOOKS = {}

_OPS_MAPPING_HOOKS = {}

_OUTPUT_MAPPING_HOOKS = {}

_PRE_FLATTEN_HOOKS = {}

# Registry for dvm ops: maps op_name -> payload_json
# todo(lmy) remove dvm op when ms_inferrt backend is ready
_DVM_OP_REGISTRY = {}


def register_dvm_op(op_name: str, payload_json: str):
    """Register a dvm op with its JSON payload.

    Args:
        op_name: The operator name (e.g., "dvm_add")
        payload_json: The DVM JSON payload
    """
    _DVM_OP_REGISTRY[op_name] = payload_json


def get_dvm_payload(op_name: str) -> str:
    """Get JSON payload for a registered dvm op."""
    return _DVM_OP_REGISTRY.get(op_name)


def register_arg_mapping_hook(op, hook_func):
    _ARG_MAPPING_HOOKS[op] = hook_func


def get_arg_mapping_hook(op):
    return _ARG_MAPPING_HOOKS.get(op)


def register_ops_mapping_hook(op, hook_func):
    _OPS_MAPPING_HOOKS[op] = hook_func


def get_ops_mapping_hook(op):
    return _OPS_MAPPING_HOOKS.get(op)


def register_output_mapping_hook(op, hook_func):
    _OUTPUT_MAPPING_HOOKS[op] = hook_func


def get_output_mapping_hook(op):
    return _OUTPUT_MAPPING_HOOKS.get(op)


def register_pre_flatten_hook(op, hook_func):
    _PRE_FLATTEN_HOOKS[op] = hook_func


def get_pre_flatten_hook(op):
    return _PRE_FLATTEN_HOOKS.get(op)


def _is_scalar_arg(arg):
    """Check if the argument is a scalar type (int, float, bool, torch.SymInt)."""
    if isinstance(arg, (int, float, bool, torch.SymInt)):
        return True
    if isinstance(arg, Node):
        if arg.target == "item":
            return True
        if isinstance(arg.meta.get("example_value", None), (int, float, bool, torch.SymInt)):
            return True
    return False


def binary_scalar_pre_flatten_hook(node):
    """Pre-flatten hook to swap scalar and tensor arguments before schema matching.
    
    For operations like add and mul, when the first argument is a scalar and the 
    second is a tensor (e.g., 2 + x), swap them to match the expected schema 
    (tensor, scalar) order. This ensures correct schema matching in _flatten_args.
    """
    if _is_scalar_arg(node.args[0]) and not _is_scalar_arg(node.args[1]):
        new_args = (node.args[1], node.args[0]) + node.args[2:]
        print(f"Pre-flatten hook: swapping args for {node.target}, "
              f"old args: {node.args}, new args: {new_args}")
        return new_args
    return node.args


# pylint: disable=unused-argument
def embedding_hook(node, input_nodes, executor):
    """Normalize embedding param order to backend (weight, indices).
    F.embedding(indices, weight) -> swap to (weight, indices).
    aten.embedding.default(weight, indices, ...) -> no swap, strip extras.
    """
    target_name = str(node.target)
    if "aten.embedding" in target_name:
        return input_nodes[:2]
    return [input_nodes[1], input_nodes[0]]


# pylint: disable=unused-argument
def binary_scalar_order_hook(node, input_nodes, executor):
    """Handle binary operations where argument order must be preserved.

    For operators like sub and div, swapping scalar and tensor arguments
    would produce incorrect results, so the (scalar, tensor) order is
    not supported.
    """
    if _is_scalar_arg(node.args[0]) and not _is_scalar_arg(node.args[1]):
        raise NotImplementedError(
            f"Operation '{node.target}' does not support (scalar, tensor) "
            f"argument order: got {type(node.args[0]).__name__} and "
            f"{type(node.args[1]).__name__}"
        )
    return input_nodes


# pylint: disable=unused-argument
def apply_rotary_pos_emb_hook(node, input_nodes, executor):
    """add layout parameter."""
    rope_layout_bsnd = 1
    return [
        input_nodes[0],
        input_nodes[1],
        input_nodes[2],
        input_nodes[3],
        rope_layout_bsnd,
    ]


# pylint: disable=unused-argument
def moe_gating_top_k_hook(node, input_nodes, executor):
    """Normalize moe_gating_top_k inputs to backend order [x, bias, k, ...]."""
    kwargs = node.kwargs

    # Canonical target order in Mrt_MoeGatingTopKOp:
    # [x, bias, k, k_group, group_count, group_select_mode, renorm, norm_type,
    #  out_flag, routed_scaling_factor, eps]
    if len(node.args) > 0:
        x = node.args[0]
    else:
        x = input_nodes[0]

    # Handle both npu frontend ("bias") and _C_ascend frontend ("bias_opt").
    bias = kwargs.get("bias", kwargs.get("bias_opt", None))
    if bias is None and len(node.args) > 2:
        bias = node.args[2]

    if "k" in kwargs:
        k = kwargs["k"]
    elif len(node.args) > 1:
        k = node.args[1]
    else:
        k = 1

    k_group = kwargs.get("k_group", 1)
    group_count = kwargs.get("group_count", 1)
    group_select_mode = kwargs.get("group_select_mode", 0)
    renorm = kwargs.get("renorm", 0)
    norm_type = kwargs.get("norm_type", 0)
    # Mrt_MoeGatingTopKOp / aclnn expect Bool for out_flag, not int.
    raw_out_flag = kwargs.get("out_flag", False)
    if isinstance(raw_out_flag, Node):
        raw_out_flag = _resolve_scalar_arg(raw_out_flag, "out_flag")
    out_flag = bool(raw_out_flag)
    routed_scaling_factor = kwargs.get("routed_scaling_factor", 1.0)
    eps = kwargs.get("eps", 1e-20)

    return [
        x,
        bias,
        k,
        k_group,
        group_count,
        group_select_mode,
        renorm,
        norm_type,
        out_flag,
        routed_scaling_factor,
        eps,
    ]


# pylint: disable=unused-argument
def moe_distribute_combine_v2_hook(node, input_nodes, executor):
    """Normalize npu_moe_distribute_combine_v2 args to backend schema order."""
    kwargs = node.kwargs
    args = node.args

    def _kw_or_pos(name, pos, default):
        if name in kwargs:
            return kwargs[name]
        if len(args) > pos:
            return args[pos]
        return default

    expand_x = _kw_or_pos("expand_x", 0, input_nodes[0] if len(input_nodes) > 0 else None)
    expert_ids = _kw_or_pos("expert_ids", 1, input_nodes[1] if len(input_nodes) > 1 else None)
    assist_info_for_combine = _kw_or_pos(
        "assist_info_for_combine", 2, input_nodes[2] if len(input_nodes) > 2 else None
    )
    ep_send_counts = _kw_or_pos("ep_send_counts", 3, input_nodes[3] if len(input_nodes) > 3 else None)
    expert_scales = _kw_or_pos("expert_scales", 4, input_nodes[4] if len(input_nodes) > 4 else None)
    group_ep = _kw_or_pos("group_ep", 5, "")
    ep_world_size = _kw_or_pos("ep_world_size", 6, 0)
    ep_rank_id = _kw_or_pos("ep_rank_id", 7, 0)
    moe_expert_num = _kw_or_pos("moe_expert_num", 8, 0)
    tp_send_counts = _kw_or_pos("tp_send_counts", 9, None)
    x_active_mask = _kw_or_pos("x_active_mask", 10, None)
    expand_scales = _kw_or_pos("expand_scales", 11, None)
    shared_expert_x = _kw_or_pos("shared_expert_x", 12, None)
    elastic_info = _kw_or_pos("elastic_info", 13, None)
    ori_x = _kw_or_pos("ori_x", 14, None)
    const_expert_alpha_1 = _kw_or_pos("const_expert_alpha_1", 15, None)
    const_expert_alpha_2 = _kw_or_pos("const_expert_alpha_2", 16, None)
    const_expert_v = _kw_or_pos("const_expert_v", 17, None)
    performance_info = _kw_or_pos("performance_info", 18, None)
    group_tp = _kw_or_pos("group_tp", 19, "")
    tp_world_size = _kw_or_pos("tp_world_size", 20, 0)
    tp_rank_id = _kw_or_pos("tp_rank_id", 21, 0)
    expert_shard_type = _kw_or_pos("expert_shard_type", 22, 0)
    shared_expert_num = _kw_or_pos("shared_expert_num", 23, 1)
    shared_expert_rank_num = _kw_or_pos("shared_expert_rank_num", 24, 0)
    global_bs = _kw_or_pos("global_bs", 25, 0)
    comm_quant_mode = _kw_or_pos("comm_quant_mode", 26, 0)
    comm_alg = _kw_or_pos("comm_alg", 27, "")
    zero_expert_num = _kw_or_pos("zero_expert_num", 28, 0)
    copy_expert_num = _kw_or_pos("copy_expert_num", 29, 0)
    const_expert_num = _kw_or_pos("const_expert_num", 30, 0)

    return [
        expand_x,
        expert_ids,
        assist_info_for_combine,
        ep_send_counts,
        expert_scales,
        tp_send_counts,
        x_active_mask,
        expand_scales,
        shared_expert_x,
        elastic_info,
        ori_x,
        const_expert_alpha_1,
        const_expert_alpha_2,
        const_expert_v,
        performance_info,
        group_ep,
        ep_world_size,
        ep_rank_id,
        moe_expert_num,
        group_tp,
        tp_world_size,
        tp_rank_id,
        expert_shard_type,
        shared_expert_num,
        shared_expert_rank_num,
        global_bs,
        comm_quant_mode,
        comm_alg,
        zero_expert_num,
        copy_expert_num,
        const_expert_num,
    ]


# pylint: disable=unused-argument
def moe_distribute_dispatch_v2_hook(node, input_nodes, executor):
    """Normalize npu_moe_distribute_dispatch_v2 args to backend schema order."""
    kwargs = node.kwargs
    args = node.args

    def _kw_or_pos(name, pos, default):
        if name in kwargs:
            return kwargs[name]
        if len(args) > pos:
            return args[pos]
        return default

    x = _kw_or_pos("x", 0, input_nodes[0] if len(input_nodes) > 0 else None)
    expert_ids = _kw_or_pos("expert_ids", 1, input_nodes[1] if len(input_nodes) > 1 else None)
    scales = _kw_or_pos("scales", 2, None)
    x_active_mask = _kw_or_pos("x_active_mask", 3, None)
    expert_scales = _kw_or_pos("expert_scales", 4, None)
    elastic_info = _kw_or_pos("elastic_info", 5, None)
    performance_info = _kw_or_pos("performance_info", 6, None)
    group_ep = _kw_or_pos("group_ep", 7, "")
    ep_world_size = _kw_or_pos("ep_world_size", 8, 0)
    ep_rank_id = _kw_or_pos("ep_rank_id", 9, 0)
    moe_expert_num = _kw_or_pos("moe_expert_num", 10, 0)
    group_tp = _kw_or_pos("group_tp", 11, "")
    tp_world_size = _kw_or_pos("tp_world_size", 12, 0)
    tp_rank_id = _kw_or_pos("tp_rank_id", 13, 0)
    expert_shard_type = _kw_or_pos("expert_shard_type", 14, 0)
    shared_expert_num = _kw_or_pos("shared_expert_num", 15, 1)
    shared_expert_rank_num = _kw_or_pos("shared_expert_rank_num", 16, 0)
    quant_mode = _kw_or_pos("quant_mode", 17, 0)
    global_bs = _kw_or_pos("global_bs", 18, 0)
    expert_token_nums_type = _kw_or_pos("expert_token_nums_type", 19, 0)
    comm_alg = _kw_or_pos("comm_alg", 20, "")
    zero_expert_num = _kw_or_pos("zero_expert_num", 21, 0)
    copy_expert_num = _kw_or_pos("copy_expert_num", 22, 0)
    const_expert_num = _kw_or_pos("const_expert_num", 23, 0)

    return [
        x,
        expert_ids,
        scales,
        x_active_mask,
        expert_scales,
        elastic_info,
        performance_info,
        group_ep,
        ep_world_size,
        ep_rank_id,
        moe_expert_num,
        group_tp,
        tp_world_size,
        tp_rank_id,
        expert_shard_type,
        shared_expert_num,
        shared_expert_rank_num,
        quant_mode,
        global_bs,
        expert_token_nums_type,
        comm_alg,
        zero_expert_num,
        copy_expert_num,
        const_expert_num,
    ]


# pylint: disable=unused-argument
def div_mod_arg_hook(node, input_nodes, executor):
    """add div mode parameter."""
    if _is_scalar_arg(node.args[0]) and not _is_scalar_arg(node.args[1]):
        raise NotImplementedError(
            f"Operation '{node.target}' does not support (scalar, tensor) "
            f"argument order: got {type(node.args[0]).__name__} and "
            f"{type(node.args[1]).__name__}"
        )
    # Built-in div_mod: tensor-tensor or tensor-scalar requires mod param, scalar-scalar does not.
    if not _is_scalar_arg(node.args[0]):
        div_mode = 2
        return [input_nodes[0], input_nodes[1], div_mode]
    return input_nodes


# pylint: disable=unused-argument
def clone_hook(node, input_nodes, executor):
    """input[1] not use"""
    return [input_nodes[0]]


# pylint: disable=unused-argument
def empty_like_hook(node, input_nodes, executor):
    """empty_like only needs self; output metadata carries shape, dtype, and device."""
    return [input_nodes[0]]


# pylint: disable=unused-argument
def long_hook(node, input_nodes, executor):
    """cast to int64 (long)."""
    return [input_nodes[0], torch.int64]


# pylint: disable=unused-argument
def float_hook(node, input_nodes, executor):
    """cast to float32."""
    return [input_nodes[0], torch.float32]


# pylint: disable=unused-argument
def int_hook(node, input_nodes, executor):
    """cast to int32."""
    return [input_nodes[0], torch.int32]


# pylint: disable=unused-argument
def argsort_hook(node, input_nodes, executor):
    """Normalize argsort inputs to [input, stable, dim, descending]."""
    if len(input_nodes) == 4:
        return input_nodes
    if len(input_nodes) == 3:
        # aten::argsort(Tensor self, int dim=-1, bool descending=False)
        # -> aclnnSort expects stable before dim/descending.
        return [input_nodes[0], False, input_nodes[1], input_nodes[2]]
    err_msg = f"Unsupported argsort input size: {len(input_nodes)}"
    raise ValueError(err_msg)


# pylint: disable=unused-argument
def permute_hook(node, input_nodes, executor):
    """transpose dims"""
    if node.target in ("transpose", torch.transpose, torch.ops.aten.transpose.int):
        dim_inx = list(range(0, len(input_nodes[0].meta["example_value"].shape), 1))
        dim_inx[input_nodes[1]] = input_nodes[2]
        dim_inx[input_nodes[2]] = input_nodes[1]
        return [input_nodes[0], dim_inx]
    # For .t(), only tensors <= 2-D are expected, so no explicit dimension parameters are required
    if node.target in ("t", torch.t, torch.ops.aten.t.default):
        dim = len(input_nodes[0].meta["example_value"].shape)
        if not dim <= 2:
            raise NotImplementedError(f".t() only supports tensors with <= 2 dimensions, but got {dim} dimensions")
        dim0 = 0
        dim1 = 1
        return [input_nodes[0], [dim1, dim0]]
    if node.target in ("movedim", torch.movedim, torch.ops.aten.movedim.int, torch.ops.aten.movedim.intlist):
        ndim = len(input_nodes[0].meta["example_value"].shape)

        def _normalize_dims(dims):
            if isinstance(dims, int):
                dims = [dims]
            normalized = []
            for dim in dims:
                dim = dim + ndim if dim < 0 else dim
                if dim < 0 or dim >= ndim:
                    raise IndexError(f"Dimension out of range for movedim: got {dim}, ndim={ndim}")
                normalized.append(dim)
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"Repeated dims are not allowed in movedim: {normalized}")
            return normalized

        source = _normalize_dims(input_nodes[1])
        destination = _normalize_dims(input_nodes[2])
        if len(source) != len(destination):
            raise ValueError(f"movedim expects source and destination to have the same length, got "
                             f"{len(source)} and {len(destination)}")

        dims = [dim for dim in range(ndim) if dim not in source]
        for dst, src in sorted(zip(destination, source)):
            dims.insert(dst, src)
        return [input_nodes[0], dims]
    return input_nodes


# pylint: disable=unused-argument
def fused_inter_attention_score_hook(node, input_nodes, executor):
    """swap the first and second param position."""
    return [
        input_nodes[0],
        [input_nodes[1]],
        [input_nodes[2]],
        input_nodes[3],
        input_nodes[4],
        input_nodes[5],
        input_nodes[6],
        input_nodes[7],
        input_nodes[8],
        input_nodes[9],
        input_nodes[10],
        input_nodes[11],
        input_nodes[12],
        input_nodes[13],
        input_nodes[18],
        input_nodes[19],
        input_nodes[20],
        input_nodes[14],
        input_nodes[15],
        input_nodes[16],
        input_nodes[17],
        input_nodes[21],
        input_nodes[22],
        input_nodes[23],
        input_nodes[24],
        input_nodes[25],
        input_nodes[26],
        input_nodes[27],
        input_nodes[28],
        input_nodes[29],
        input_nodes[30],
        input_nodes[31],
        input_nodes[32],
        input_nodes[33],
        input_nodes[34],
        input_nodes[35],
        input_nodes[36],
        input_nodes[39],
        input_nodes[37],
        input_nodes[38],
    ]


def _resolve_scalar_arg(arg, name: str):
    """Resolve scalar-like FX arg to python value."""
    if isinstance(arg, Node):
        arg = arg.meta.get("example_value", None)
    if arg is None:
        raise ValueError(f"Failed to resolve scalar argument '{name}'")
    return arg


# pylint: disable=unused-argument
def dequant_swiglu_quant_hook(node, input_nodes, executor):
    """Normalize npu_dequant_swiglu_quant args to aclnnDequantSwigluQuant(V1)."""
    if len(input_nodes) < 13:
        err_msg = f"Unsupported npu_dequant_swiglu_quant input size: {len(input_nodes)}"
        raise ValueError(err_msg)

    quant_mode = int(_resolve_scalar_arg(input_nodes[8], "quant_mode"))

    if quant_mode not in (0, 1):
        raise ValueError(f"quant_mode only supports 0(static) or 1(dynamic), but got {quant_mode}")

    quant_mode_str = "static" if quant_mode == 0 else "dynamic"
    return [
        input_nodes[0],  # x
        input_nodes[1],  # weight_scale
        input_nodes[2],  # activation_scale
        input_nodes[3],  # bias
        input_nodes[4],  # quant_scale
        input_nodes[5],  # quant_offset
        input_nodes[6],  # group_index
        input_nodes[7],  # activate_left
        quant_mode_str,
    ]


# pylint: disable=unused-argument
def dequant_swiglu_quant_op_hook(op, node, input_nodes, executor):
    """Fallback to custom_call when V2-only controls are used."""
    if len(input_nodes) < 13:
        return Op.custom_call
    try:
        quant_mode = int(_resolve_scalar_arg(input_nodes[8], "quant_mode"))
        swiglu_mode = int(_resolve_scalar_arg(input_nodes[9], "swiglu_mode"))
        clamp_limit = float(_resolve_scalar_arg(input_nodes[10], "clamp_limit"))
        glu_alpha = float(_resolve_scalar_arg(input_nodes[11], "glu_alpha"))
        glu_bias = float(_resolve_scalar_arg(input_nodes[12], "glu_bias"))
    except (TypeError, ValueError):
        return Op.custom_call

    if quant_mode not in (0, 1):
        return Op.custom_call

    if swiglu_mode != 0:
        return Op.custom_call

    if abs(clamp_limit - 7.0) > 1e-6 or abs(glu_alpha - 1.702) > 1e-6 or abs(glu_bias - 1.0) > 1e-6:
        return Op.custom_call

    return op


def _extract_tensor_example(arg, err_msg: str):
    """Resolve a tensor example_value from an FX node or eager value."""
    if isinstance(arg, Node):
        arg = arg.meta.get("example_value", None)
    if not isinstance(arg, torch.Tensor):
        raise RuntimeError(err_msg)
    return arg


def _add_tuple_getitem_node(executor, sym_mgr, tuple_node, index: int, output_value):
    """Project one item from a tuple-valued op result."""
    index_node = executor.add_value_node(sym_mgr.from_torch_with_sym(index))
    return executor.add_op_node(Op.tuple_getitem, [tuple_node, index_node], output_value)


def argsort_output_hook(node, op, input_nodes, executor, sym_mgr):
    """Lower argsort to its tuple output and project the indices result."""
    if not node.args:
        raise RuntimeError("argsort requires at least one input tensor")

    output_example = _extract_tensor_example(
        node.meta.get("example_value", None),
        "argsort example_value must be a tensor",
    )
    input_example = _extract_tensor_example(
        node.args[0],
        "argsort input example_value must be a tensor",
    )

    # Runtime argsort produces (values, indices), while FX argsort returns indices only.
    tuple_output = sym_mgr.from_torch_with_sym((input_example, output_example))
    tuple_node = executor.add_op_node(op, input_nodes, tuple_output)
    output_value = sym_mgr.from_torch_with_sym(output_example)
    return _add_tuple_getitem_node(executor, sym_mgr, tuple_node, 1, output_value)


def rms_norm_output_hook(node, op, input_nodes, executor, sym_mgr):
    """
    Adapt Op.rms_norm tuple output for torch.rms_norm single-tensor semantics.

    - torch.rms_norm(...) returns one Tensor.
    - Backend Op.rms_norm returns (y, rstd).

    For torch.rms_norm target, materialize tuple output in IR and project y (index 0).
    For other rms_norm targets (e.g. npu_rms_norm) keep original output shape.
    """
    target_name = getattr(node.target, "__name__", None)
    is_torch_rms_norm = target_name == "rms_norm" and getattr(node.target, "__module__", "").startswith("torch")

    example_value = node.meta.get("example_value", None)
    if not is_torch_rms_norm:
        output_value = sym_mgr.from_torch_with_sym(example_value)
        return executor.add_op_node(op, input_nodes, output_value)

    output_example = _extract_tensor_example(
        example_value,
        "rms_norm example_value must be a tensor",
    )
    x_example = _extract_tensor_example(
        node.args[0] if len(node.args) > 0 else None,
        "rms_norm input example_value must be a tensor",
    )
    gamma_example = _extract_tensor_example(
        node.args[2] if len(node.args) > 2 else None,
        "rms_norm gamma example_value must be a tensor",
    )

    rstd_dim = x_example.dim() - gamma_example.dim()
    rstd_shape = [x_example.size(i) if i < rstd_dim else 1 for i in range(x_example.dim())]
    rstd_example = output_example.new_empty(rstd_shape, dtype=torch.float32)

    tuple_output = sym_mgr.from_torch_with_sym((output_example, rstd_example))
    tuple_node = executor.add_op_node(op, input_nodes, tuple_output)
    output_value = sym_mgr.from_torch_with_sym(output_example)
    return _add_tuple_getitem_node(executor, sym_mgr, tuple_node, 0, output_value)


def moe_distribute_dispatch_v2_output_hook(node, op, input_nodes, executor, sym_mgr):
    """
    Adapt dispatch_v2 tuple output when fake/meta infers empty expand_scales.

    Some torch_npu meta paths may produce expand_scales with shape [0], while
    runtime eager returns a non-empty shape derived from expand_x and tp_world_size.
    """
    example_value = node.meta.get("example_value", None)
    if not isinstance(example_value, (tuple, list)) or len(example_value) != 7:
        output_value = sym_mgr.from_torch_with_sym(example_value)
        return executor.add_op_node(op, input_nodes, output_value)

    outputs = list(example_value)
    expand_x = outputs[0]
    expand_scales = outputs[6]

    if hasattr(expand_scales, "numel") and int(expand_scales.numel()) == 0:
        tp_world_size_arg = node.kwargs.get("tp_world_size", node.args[12] if len(node.args) > 12 else 0)
        tp_world_size_val = _get_example_value_if_node(tp_world_size_arg)

        try:
            tp_world_size = int(tp_world_size_val)
        except (TypeError, ValueError):
            tp_world_size = 0
        if tp_world_size == 0:
            tp_world_size = 1

        expand_scales_len = expand_x.shape[0] // tp_world_size
        outputs[6] = expand_scales.new_empty((expand_scales_len,), dtype=expand_scales.dtype)

    output_value = sym_mgr.from_torch_with_sym(tuple(outputs))
    return executor.add_op_node(op, input_nodes, output_value)


def _init_arg_mapping_hooks():
    """register hooks for mapping input arguments"""
    register_arg_mapping_hook(Op.clone, clone_hook)
    register_arg_mapping_hook(Op.empty_like, empty_like_hook)
    register_arg_mapping_hook(Op.argsort, argsort_hook)
    register_arg_mapping_hook(
        Op.fused_infer_attention_score, fused_inter_attention_score_hook
    )
    register_arg_mapping_hook(Op.permute, permute_hook)
    register_arg_mapping_hook(Op.permute_view, permute_hook)
    register_arg_mapping_hook(Op.embedding, embedding_hook)
    register_arg_mapping_hook(Op.sub_scalar, binary_scalar_order_hook)
    register_arg_mapping_hook(Op.div_scalar, binary_scalar_order_hook)
    register_arg_mapping_hook(Op.div_mod_scalar, div_mod_arg_hook)
    register_arg_mapping_hook(Op.apply_rotary_pos_emb, apply_rotary_pos_emb_hook)
    register_arg_mapping_hook(Op.moe_gating_top_k, moe_gating_top_k_hook)
    register_arg_mapping_hook(Op.moe_distribute_combine_v2, moe_distribute_combine_v2_hook)
    register_arg_mapping_hook(Op.moe_distribute_dispatch_v2, moe_distribute_dispatch_v2_hook)
    register_arg_mapping_hook(Op.dequant_swiglu_quant, dequant_swiglu_quant_hook)
    register_arg_mapping_hook(Op.reduce_sum, reduce_sum_arg_hook)
    register_arg_mapping_hook(Op.squeeze_view, squeeze_arg_hook)
    register_arg_mapping_hook(Op.reduce_mean, reduce_sum_arg_hook)
    register_arg_mapping_hook(Op.index_tensor, index_tensor_arg_hook)
    register_arg_mapping_hook(Op.amax, amax_arg_hook)
    register_arg_mapping_hook(Op.var_mean, var_mean_arg_hook)
    # dtype cast-style tensor methods
    register_arg_mapping_hook("long", long_hook)
    register_arg_mapping_hook("float", float_hook)
    register_arg_mapping_hook("int", int_hook)
    # chunk lowering
    register_arg_mapping_hook(torch.chunk, chunk_arg_hook)
    register_arg_mapping_hook(aten.chunk.default, chunk_arg_hook)
    register_arg_mapping_hook("chunk", chunk_arg_hook)
    # in-place index_put_: always materialize both accumulate and unsafe,
    # defaulting to False when omitted by the frontend.
    register_arg_mapping_hook(Op.index_put, index_put_arg_hook)
    # Normalize torch.rms_norm argument layout to backend Op.rms_norm layout.
    register_arg_mapping_hook(Op.rms_norm, rms_norm_arg_hook)
    register_arg_mapping_hook(Op.arange, arange_arg_hook)
    register_arg_mapping_hook(Op.iota, iota_arg_hook)
    register_arg_mapping_hook(Op.leaky_relu, leaky_relu_arg_hook)
    register_arg_mapping_hook(Op.log_softmax, log_softmax_arg_hook)
    # Normalize torch.layer_norm / torch.nn.functional.layer_norm to backend Op.norm layout.
    register_arg_mapping_hook(Op.norm, layer_norm_arg_hook)


def _init_pre_flatten_hooks():
    """register hooks for pre-flatten argument adjustment"""
    register_pre_flatten_hook(Op.add_scalar, binary_scalar_pre_flatten_hook)
    register_pre_flatten_hook(Op.mul_scalar, binary_scalar_pre_flatten_hook)
    register_pre_flatten_hook(Op.eq_scalar, binary_scalar_pre_flatten_hook)


def leaky_relu_arg_hook(node, flat_args, executor):
    """Ensure negative_slope defaults to 0.01 when omitted for leaky_relu."""
    args = list(flat_args)
    if len(args) < 2:
        args.append(0.01)
    return args


def log_softmax_arg_hook(node, flat_args, executor):
    """
    Normalize log_softmax frontend variants to backend Op.log_softmax layout.

    ACLNN consumes only [self, dim]. PyTorch's dtype / half_to_float
    arguments also change the input computation dtype, so reject promotion
    until an explicit cast is lowered before log_softmax.
    """
    args = list(flat_args)
    if len(args) < 2:
        raise ValueError(f"Unexpected log_softmax argument count after schema flatten: {len(args)}")

    if node.target == aten._log_softmax.default:  # pylint: disable=protected-access
        half_to_float = args[2] if len(args) > 2 else False
        if half_to_float:
            raise ValueError(
                "log_softmax half_to_float=True requires casting the input tensor first, "
                "which is not supported yet"
            )
        return args[:2]

    dtype_arg = args[2] if len(args) > 2 else None
    if dtype_arg is not None:
        input_dtype = get_tensor_arg_dtype(args[0])
        if input_dtype is not None and dtype_arg != input_dtype:
            raise ValueError(
                f"log_softmax dtype={dtype_arg} requires casting the input tensor from "
                f"{input_dtype} first, which is not supported yet"
            )
    return args[:2]


def amax_arg_hook(node, flat_args, executor):
    """
    Normalize amax arguments.

    PyTorch uses dim=[] as all-dimension reduction. aclnnAmax expects the
    concrete dimension list, so expand [] to [0, ..., rank - 1].
    """
    args = list(flat_args)
    if len(args) < 3:
        raise ValueError(f"Unexpected amax argument count after schema flatten: {len(args)}")

    self_arg = args[0]
    dim = args[1]
    example = self_arg.meta.get("example_value", None) if isinstance(self_arg, Node) else self_arg

    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        rank = int(example.dim()) if hasattr(example, "dim") else 0
        args[1] = list(range(rank))
    elif isinstance(dim, int):
        args[1] = [dim]

    return args[:3]


def index_put_arg_hook(node, flat_args, executor):
    """
    Normalize arguments for index_put_ / aten.index_put_.default.

    aten.index_put_.default schema (simplified):
      self, indices, values, accumulate=False, unsafe=False
    """
    # flat_args: [self, indices, values, (accumulate)?, (unsafe)?]
    args = list(flat_args)

    # Ensure accumulate exists (position 3)
    if len(args) < 4:
        args.append(False)

    # Ensure unsafe exists (position 4)
    if len(args) < 5:
        args.append(False)

    return args


# pylint: disable=unused-argument
def index_tensor_arg_hook(node, flat_args, executor):
    """
    Normalize aten.index.Tensor indices by removing None placeholders.

    aten.index.Tensor schema:
      self, indices
    """
    args = list(flat_args)
    if len(args) < 2:
        return args

    indices = args[1]
    if isinstance(indices, (list, tuple)):
        normalized_indices = list(indices)
        while normalized_indices and normalized_indices[-1] is None:
            normalized_indices.pop()
        args[1] = normalized_indices

    return args


# pylint: disable=unused-argument
def arange_arg_hook(node, flat_args, executor):
    """Normalize torch.arange arguments based on original call signature.

    Handles:
      - arange(end, **kw)
      - arange(start, end, **kw)
      - arange(start, end, step, **kw)
    Ensures step defaults to 1 and returns (start, end, step, dtype?, device?, ...).
    """
    orig_args = node.args
    orig_kwargs = node.kwargs

    if len(orig_args) == 1:
        # torch.arange(end, **kwargs)
        start, end, step = 0, orig_args[0], orig_kwargs.get("step", 1)
    elif len(orig_args) == 2:
        # torch.arange(start, end, **kwargs)
        start, end, step = orig_args[0], orig_args[1], orig_kwargs.get("step", 1)
    elif len(orig_args) >= 3:
        # torch.arange(start, end, step, **kwargs)
        start, end, step = orig_args[0], orig_args[1], orig_args[2]
    else:
        return list(flat_args)

    if step is None:
        step = 1

    new_args = [start, end, step]
    # Preserve dtype/device and other kwargs that were explicitly passed
    for k in ("dtype", "device", "out", "layout", "pin_memory"):
        if k in orig_kwargs and orig_kwargs[k] is not None:
            new_args.append(orig_kwargs[k])

    return new_args


def iota_arg_hook(node, flat_args, executor):
    """
    Normalize prims.iota arguments to backend Op.iota layout.

    The torch schema is:
      prims.iota(length, *, start, step, dtype, device, requires_grad)
    Op.iota consumes only length/start/step and gets output dtype/device from
    torch meta output. requires_grad=True is not supported.
    """
    args = list(flat_args)
    if len(args) < 6:
        raise ValueError(f"Unexpected prims.iota argument count after schema flatten: {len(args)}")

    requires_grad = args[5]
    if requires_grad:
        raise ValueError("prims.iota requires_grad=True is not supported")

    return args[:3]


def rms_norm_arg_hook(node, flat_args, executor):
    """
    Normalize arguments for rms_norm.

    Supported input forms:
    - torch.rms_norm(x, normalized_shape, weight, eps)
    - torch.ops.npu.npu_rms_norm(x, gamma, epsilon)

    Backend Op.rms_norm expects exactly:
    [x, gamma, epsilon]
    """
    args = list(flat_args)
    # torch.rms_norm path: [x, normalized_shape, weight, eps]
    if len(args) >= 4:
        x = args[0]
        gamma = args[2]
        epsilon = args[3]
        return [x, gamma, epsilon]
    # Already in expected form (e.g. npu_rms_norm)
    if len(args) == 3:
        return args
    # Fallback: keep original args for better runtime diagnostics
    return args


def layer_norm_arg_hook(node, flat_args, executor):
    """
    Normalize torch.nn.functional.layer_norm arguments to backend Op.norm layout.

    torch.nn.functional.layer_norm is mapped to torch.layer_norm for schema lookup.
    aten::layer_norm has cudnn_enable as the sixth argument, but NPU does not
    support or consume it. Backend Op.norm expects exactly:
    [x, normalized_shape, weight, bias, eps]
    """
    args = list(flat_args)
    if len(args) >= 6:
        return args[:5]
    if len(args) == 5:
        return args
    raise ValueError(f"Unexpected layer_norm argument count after schema flatten: {len(args)}")


def _get_chunk_example_outputs(node):
    """Return the list of chunk outputs from FX node meta, or None if missing."""
    example_value = node.meta.get("example_value", None)
    if not isinstance(example_value, (tuple, list)) or not example_value:
        return None
    first = example_value[0]
    if not hasattr(first, "shape"):
        return None
    return example_value


def _resolve_chunk_dim(dim_arg, input_tensor, ref_tensor):
    """Resolve the dim argument for chunk to a concrete integer or None.

    dim_arg can be:
      - a Python int or torch.SymInt
      - an FX Node whose meta['example_value'] is an int
      - or something else, in which case we try to infer from shapes
    """
    # Direct integer or SymInt
    if isinstance(dim_arg, (int, torch.SymInt)):
        return int(dim_arg)

    # dim passed as FX Node with example_value
    if isinstance(dim_arg, Node):
        dim_example = dim_arg.meta.get("example_value", None)
        if isinstance(dim_example, (int, torch.SymInt)):
            return int(dim_example)

    # Fallback: infer from input/output shapes
    if isinstance(input_tensor, Node):
        input_example = input_tensor.meta.get("example_value", None)
    else:
        input_example = input_tensor

    if hasattr(input_example, "shape") and hasattr(ref_tensor, "shape"):
        in_shape = tuple(input_example.shape)
        out_shape = tuple(ref_tensor.shape)
        if len(in_shape) == len(out_shape):
            for i, (si, so) in enumerate(zip(in_shape, out_shape)):
                if si != so:
                    return i

    return None


# pylint: disable=unused-argument
def chunk_arg_hook(node, input_nodes, executor):
    """Lower torch.chunk to split_with_size by constructing split_sizes.

    We do not have a dedicated `chunk` op in the runtime. Instead, we use
    the FX node's example_value (a tuple/list of chunk tensors) to derive
    per-chunk sizes along the given dim and call the existing split_with_size
    implementation. This guarantees we match PyTorch's chunk behavior.
    """
    # Example outputs from FX (tuple/list of chunk tensors)
    example_value = _get_chunk_example_outputs(node)
    if example_value is None:
        return input_nodes
    ref_tensor = example_value[0]

    args = list(node.args)
    kwargs = dict(node.kwargs)

    # Input tensor (self) – forward original FX argument for later mapping.
    if not args:
        return input_nodes
    input_tensor = args[0]

    # Resolve dim to a concrete integer.
    dim_arg = 0
    if len(args) >= 3:
        dim_arg = args[2]
    elif "dim" in kwargs:
        dim_arg = kwargs["dim"]

    dim_int = _resolve_chunk_dim(dim_arg, input_tensor, ref_tensor)
    if dim_int is None:
        return input_nodes

    rank = ref_tensor.dim()
    if dim_int < 0:
        dim_int += rank

    # Derive per-chunk sizes along `dim` directly from example_value. Keep
    # SymInt sizes symbolic; int(SymInt) would add value guards and specialize
    # dynamic chunk lengths to the first example input.
    split_sizes = [t.shape[dim_int] for t in example_value]

    return [input_tensor, split_sizes, dim_int]


# pylint: disable=unused-argument
def squeeze_arg_hook(node, flat_args, executor):
    """Normalize squeeze default/dim/dims forms to backend [self, dims]."""
    args = list(flat_args)
    if not args:
        return args

    self_arg = args[0]
    dim = args[1] if len(args) > 1 else None
    if dim is None:
        example = self_arg.meta.get("example_value", None) if isinstance(self_arg, Node) else self_arg
        rank = int(example.dim()) if hasattr(example, "dim") else 0
        dims = list(range(rank))
    elif isinstance(dim, (list, tuple)):
        dims = list(dim)
    else:
        dims = [dim]

    return [self_arg, dims]


# pylint: disable=unused-argument
def reduce_sum_arg_hook(node, flat_args, executor):
    """
    Normalize arguments for reduce_sum / sum:
    - dim=None or [] -> all dimensions [0..rank-1]
    - keepdim: use schema default
    - dtype=None -> use input dtype (matches torch semantics when dtype is not specified)
    flat_args layout (from aten::sum.dim_IntList schema):
      [self, dim, keepdim, dtype]
    """

    # Unpack with safe defaults in case of unexpected arity
    self_arg = flat_args[0] if len(flat_args) > 0 else None
    dim = flat_args[1] if len(flat_args) > 1 else None
    keepdim = flat_args[2] if len(flat_args) > 2 else False
    dtype = flat_args[3] if len(flat_args) > 3 else None

    # Get example tensor to infer rank / dtype
    example = None
    if isinstance(self_arg, Node):
        example = self_arg.meta.get("example_value", None)
    else:
        example = self_arg

    # Normalize dim: None or empty list -> reduce over all dims
    if dim is None:
        if hasattr(example, "dim"):
            try:
                rank = int(example.dim())
                dims = list(range(rank))
            except Exception:  # pylint: disable=broad-exception-caught
                dims = []
        else:
            dims = []
    elif isinstance(dim, (list, tuple)) and len(dim) == 0:
        if hasattr(example, "dim"):
            try:
                rank = int(example.dim())
                dims = list(range(rank))
            except Exception:  # pylint: disable=broad-exception-caught
                dims = []
        else:
            dims = []
    else:
        # Backend expects dimensions as Tuple (I64Array); single int/SymInt/Node -> [dim]
        if isinstance(dim, (list, tuple)):
            dims = dim
        else:
            dims = [dim]

    # Normalize dtype: None -> use input dtype (Tensor or FakeTensor-like)
    if dtype is None:
        # Typical case: real Tensor
        if isinstance(example, torch.Tensor):
            dtype = example.dtype
        # FakeTensor or other tensor-like with dtype attribute
        elif hasattr(example, "dtype"):
            try:
                dtype = example.dtype
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    return [self_arg, dims, keepdim, dtype]


# pylint: disable=unused-argument
def var_mean_arg_hook(node, flat_args, executor):
    """Normalize arguments for var_mean to backend schema [input, dim, correction, keepdim]."""
    args = list(flat_args)
    kwargs = node.kwargs if node is not None else {}

    input_arg = args[0] if args else None
    dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
    correction = kwargs.get("correction", args[2] if len(args) > 2 else None)
    keepdim = kwargs.get("keepdim", args[3] if len(args) > 3 else None)

    if correction is None:
        correction = 1
    if keepdim is None:
        keepdim = False

    # Normalize dim: None or [] -> all dims; scalar -> wrap in list
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        example = input_arg.meta.get("example_value", None) if isinstance(input_arg, Node) else input_arg
        if hasattr(example, "dim"):
            try:
                dim = list(range(int(example.dim())))
            except Exception:  # pylint: disable=broad-exception-caught
                dim = []
        else:
            dim = []
    elif not isinstance(dim, (list, tuple)):
        dim = [dim]

    return [input_arg, list(dim), int(correction), bool(keepdim)]


# pylint: disable=unused-argument
def split_ops_hook(op, node, input_nodes, executor):
    """
    Hook to determine which split op to use for a given FX node.

    If the target is torch.chunk or the string "chunk", preserves the mapped op.
    If the second input node is an integer or torch.SymInt, returns Op.split_tensor_view.
    If the second input node has a 'meta' attribute whose 'example_value'
    is an integer or torch.SymInt, returns Op.split_tensor_view.
    Otherwise, returns the original op.
    """
    if _is_chunk_target(node.target):
        return op

    if isinstance(input_nodes[1], (int, torch.SymInt)):
        return Op.split_tensor_view
    if hasattr(input_nodes[1], "meta") and input_nodes[1].meta is not None:
        if isinstance(
                input_nodes[1].meta.get("example_value", None), (int, torch.SymInt)
        ):
            return Op.split_tensor_view
    return op


# pylint: disable=unused-argument
def masked_fill_op_hook(op, node, input_nodes, executor):
    """Get the masked_fill op for a given node."""
    if isinstance(node.args[-1], (int, float)):
        return Op.masked_fill_scalar
    return Op.masked_fill_tensor


# pylint: disable=unused-argument
def inplace_masked_fill_op_hook(op, node, input_nodes, executor):
    """Get the inplace_masked_fill op for a given node."""
    if isinstance(node.args[-1], (int, float)):
        return Op.inplace_masked_fill_scalar
    return Op.inplace_masked_fill_tensor


# pylint: disable=unused-argument
def ge_op_hook(op, node, input_nodes, executor):
    """Get the ge op for a given node."""
    if _is_scalar_arg(node.args[-1]):
        return Op.ge_scalar
    return Op.ge


# pylint: disable=unused-argument
def lt_op_hook(op, node, input_nodes, executor):
    """Get the lt op for a given node."""
    if _is_scalar_arg(node.args[-1]):
        return Op.lt_scalar
    return Op.lt


# pylint: disable=unused-argument
def le_op_hook(op, node, input_nodes, executor):
    """Get the le op for a given node."""
    if _is_scalar_arg(node.args[-1]):
        return Op.le_scalar
    return Op.le


# pylint: disable=unused-argument
def add_op_hook(op, node, input_nodes, executor):
    """Get the add op for a given node."""
    if _is_scalar_arg(node.args[0]) or _is_scalar_arg(node.args[1]):
        return Op.add_scalar
    return Op.add


# pylint: disable=unused-argument
def sub_op_hook(op, node, input_nodes, executor):
    """Get the sub op for a given node."""
    if _is_scalar_arg(node.args[0]) or _is_scalar_arg(node.args[1]):
        return Op.sub_scalar
    return Op.sub


# pylint: disable=unused-argument
def mul_op_hook(op, node, input_nodes, executor):
    """Get the mul op for a given node."""
    if _is_scalar_arg(node.args[0]) or _is_scalar_arg(node.args[1]):
        return Op.mul_scalar
    return Op.mul


# pylint: disable=unused-argument
def div_op_hook(op, node, input_nodes, executor):
    """Get the div op for a given node."""
    if _is_scalar_arg(node.args[0]) or _is_scalar_arg(node.args[1]):
        return Op.div_scalar
    return Op.div

# pylint: disable=unused-argument
def mod_op_hook(op, node, input_nodes, executor):
    """Get the div op for a given node."""
    if _is_scalar_arg(node.args[1]):
        return Op.remainder_tensor_scalar
    return Op.remainder_tensor_tensor

# pylint: disable=unused-argument
def div_mod_op_hook(op, node, input_nodes, executor):
    """Get the div_mod op for a given node."""
    if _is_scalar_arg(node.args[0]) or _is_scalar_arg(node.args[1]):
        return Op.div_mod_scalar
    return Op.div_mod


# pylint: disable=unused-argument
def inplace_add_op_hook(op, node, input_nodes, executor):
    """Get the inplace_add op for a given node."""
    if isinstance(node.args[1], (int, float, bool)):
        return Op.inplace_add_scalar
    return Op.inplace_add


# pylint: disable=unused-argument
def copy_op_hook(op, node, input_nodes, executor):
    """Get the copy op for a given node."""
    if _is_scalar_arg(node.args[-1]):
        return Op.inplace_fill_scalar
    return Op.inplace_copy


# pylint: disable=unused-argument
def fill_op_hook(op, node, input_nodes, executor):
    """Get the fill op for a given node."""
    if _is_scalar_arg(node.args[-1]):
        return Op.inplace_fill_scalar
    return Op.inplace_fill_tensor
def eq_func_hook(op, node, input_nodes, executor):
    """Get the eq op for torch.eq function - detects scalar vs tensor."""
    if _is_scalar_arg(node.args[0]) or _is_scalar_arg(node.args[1]):
        return Op.eq_scalar
    return Op.eq

def _init_ops_mapping_hooks():
    """Register ops mapping hooks for torch ops."""
    register_ops_mapping_hook(Op.split_with_size_view, split_ops_hook)
    register_ops_mapping_hook(Op.masked_fill_tensor, masked_fill_op_hook)
    register_ops_mapping_hook(
        Op.inplace_masked_fill_tensor, inplace_masked_fill_op_hook
    )
    register_ops_mapping_hook(Op.inplace_fill_tensor, fill_op_hook)
    register_ops_mapping_hook(Op.inplace_copy, copy_op_hook)
    register_ops_mapping_hook(Op.ge, ge_op_hook)
    register_ops_mapping_hook(Op.lt, lt_op_hook)
    register_ops_mapping_hook(Op.eq, eq_func_hook)
    register_ops_mapping_hook(Op.le, le_op_hook)
    register_ops_mapping_hook(Op.add, add_op_hook)
    register_ops_mapping_hook(Op.sub, sub_op_hook)
    register_ops_mapping_hook(Op.mul, mul_op_hook)
    register_ops_mapping_hook(Op.div, div_op_hook)
    register_ops_mapping_hook(Op.div_mod, div_mod_op_hook)
    register_ops_mapping_hook(Op.remainder_tensor_tensor, mod_op_hook)
    register_ops_mapping_hook(Op.inplace_add, inplace_add_op_hook)
    register_ops_mapping_hook(Op.dequant_swiglu_quant, dequant_swiglu_quant_op_hook)


def _init_output_mapping_hooks():
    """Register output mapping hooks for runtime ops."""
    register_output_mapping_hook(Op.argsort, argsort_output_hook)
    register_output_mapping_hook(Op.rms_norm, rms_norm_output_hook)
    register_output_mapping_hook(Op.moe_distribute_dispatch_v2, moe_distribute_dispatch_v2_output_hook)


def _next_unique_graph_id():
    global _GLOBAL_GRAPH_ID
    _GLOBAL_GRAPH_ID += 1
    return _GLOBAL_GRAPH_ID


def _match_node_by_name(node, op_type, name):
    """Check if a node matches the given op type and name."""
    if node.op != op_type:
        return False
    if op_type == "call_function" and hasattr(node.target, "__name__"):
        return node.target.__name__ == name
    if op_type == "call_method" and isinstance(node.target, str):
        return node.target == name
    return False


# Remove unwanted nodes before processing
def _remove_matched_nodes(gm: GraphModule, matchers):
    """
    Remove nodes from the graph that match any of the given matchers.

    Args:
        gm: The GraphModule to process.
        matchers: A list of matcher specifications. Each matcher can be:
            - A tuple of (op_type, name): matches nodes by op type and name

    Returns:
        int: The number of nodes removed.
    """
    nodes_to_erase = []
    for node in gm.graph.nodes:
        for matcher in matchers:
            if isinstance(matcher, tuple) and len(matcher) == 2:
                op_type, name = matcher
                should_remove = _match_node_by_name(node, op_type, name)
                if should_remove:
                    nodes_to_erase.append(node)

    for node in nodes_to_erase:
        gm.graph.erase_node(node)

    # Recompile GraphModule after graph modification to keep internal state consistent
    if nodes_to_erase:
        gm.recompile()


aten = torch.ops.aten
# pylint: disable=protected-access
# A comprehensive mapping from torch fx ops to our custom ops.
_OP_MAP = {
    # torch functions
    torch.add: Op.add,
    torch.argsort: Op.argsort,
    torch.sub: Op.sub,
    torch.mul: Op.mul,
    torch.div: Op.div,
    torch.eq: Op.eq,
    torch.ne: Op.ne,
    torch.lt: Op.lt,
    torch.le: Op.le,
    torch.bitwise_or: Op.bitwise_or_tensor,
    torch.gt: Op.gt,
    torch.ge: Op.ge,
    torch.mm: Op.mm,
    torch.matmul: Op.matmul,
    torch.masked_fill: Op.masked_fill_tensor,
    torch.reshape: Op.view,
    torch.t: Op.permute_view,
    torch.permute: Op.permute_view,
    torch.transpose: Op.permute_view,
    torch.movedim: Op.permute_view,
    torch.squeeze: Op.squeeze_view,
    torch.unsqueeze: Op.unsqueeze_view,
    torch.narrow: Op.narrow_view,
    torch.unbind: Op.unbind_view,
    torch.split: Op.split_with_size_view,
    torch.chunk: Op.split_with_size_view,
    torch.flatten: Op.flatten_view,
    torch.cat: Op.cat,
    torch.stack: Op.stack,
    torch.sum: Op.reduce_sum,
    torch.amax: Op.amax,
    torch.mean: Op.reduce_mean,
    torch.clone: Op.clone,
    torch.index_select: Op.index_select,
    torch.neg: Op.neg,
    torch.square: Op.square,
    torch.pow: Op.pow_scalar,
    torch.rsqrt: Op.rsqrt,
    torch.exp: Op.exp,
    torch.log_softmax: Op.log_softmax,
    aten.rsqrt: Op.rsqrt,
    aten.rsqrt.default: Op.rsqrt,
    aten.exp: Op.exp,
    aten.exp.default: Op.exp,
    aten.index.Tensor: Op.index_tensor,
    aten.le.Tensor: Op.le,
    aten.le.Scalar: Op.le,
    aten.mm.default: Op.mm,
    aten.mul.Tensor: Op.mul,
    aten.neg.default: Op.neg,
    aten.sum.dim_IntList: Op.reduce_sum,
    torch.relu: Op.relu,
    torch.sigmoid: Op.sigmoid,
    torch.empty: Op.empty,
    torch.empty_like: Op.empty_like,
    torch.zeros: Op.zeros,
    torch.arange: Op.arange,
    torch.tril: Op.tril,
    torch.topk: Op.topk,
    torch.select: Op.select_view,
    torch.layer_norm: Op.norm,
    torch.ops.aten.alias.default: Op.alias,
    aten.view.default: Op.view,
    aten.reshape.default: Op.view,
    aten.permute.default: Op.permute_view,
    aten.transpose.int: Op.permute_view,
    aten.t.default: Op.permute_view,
    aten.movedim.int: Op.permute_view,
    aten.movedim.intlist: Op.permute_view,
    aten.flatten.using_ints: Op.flatten_view,
    aten.select.int: Op.select_view,
    aten.slice.Tensor: Op.slice_view,
    aten.squeeze.default: Op.squeeze_view,
    aten.squeeze.dim: Op.squeeze_view,
    aten.squeeze.dims: Op.squeeze_view,
    aten.unsqueeze.default: Op.unsqueeze_view,
    aten.narrow.default: Op.narrow_view,
    aten.unbind.int: Op.unbind_view,
    aten.chunk.default: Op.split_with_size_view,
    aten.split.Tensor: Op.split_with_size_view,
    aten.split.sizes: Op.split_with_size_view,
    aten.split.default: Op.split_with_size_view,
    aten.split_with_sizes.default: Op.split_with_size_view,
    aten.copy_.default: Op.inplace_copy,
    aten.empty_like: Op.empty_like,
    aten.empty_like.default: Op.empty_like,
    aten.expand.default: Op.expand,
    aten.add.Tensor: Op.add,
    aten.bmm.default: Op.batch_matmul,
    aten.cat.default: Op.cat,
    aten.clone.default: Op.clone,
    aten.div.Tensor: Op.div,
    aten.embedding.default: Op.embedding,
    aten.eq.Scalar: Op.eq_scalar,
    aten.eq.Tensor: Op.eq,
    aten.index_put_.default: Op.index_put,
    aten.index_copy_.default: Op.inplace_index_copy,
    aten.add_.Scalar: Op.inplace_add,
    aten.arange: Op.arange,
    aten.arange.default: Op.arange,
    aten.arange.start: Op.arange,
    torch.ops.prims.iota.default: Op.iota,
    aten.amax.default: Op.amax,
    aten.tril: Op.tril,
    aten.topk: Op.topk,
    aten.topk.default: Op.topk,
    aten.sub.Tensor: Op.sub,
    aten.var_mean.correction: Op.var_mean,
    aten.where.self: Op.where,
    torch.ops.prims.convert_element_type.default: Op.cast,
    aten.mean: Op.reduce_mean,
    aten.mean.dim: Op.reduce_mean,
    aten.pow: Op.pow_scalar,
    aten.pow.Tensor_Scalar: Op.pow_scalar,
    aten.pow.Tensor_Tensor: Op.pow_tensor,
    aten._log_softmax.default: Op.log_softmax,
    aten.log_softmax.int: Op.log_softmax,
    aten.softmax: Op.softmax,
    aten.softmax.int: Op.softmax,
    aten._softmax.default: Op.softmax,
    aten.log_softmax: Op.log_softmax,
    aten.masked_fill.Scalar: Op.masked_fill_scalar,
    aten.native_layer_norm.default: Op.norm,
    aten.silu.default: Op.silu,
    aten.stack.default: Op.stack,
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
    torch.nn.functional.leaky_relu: Op.leaky_relu,
    torch.nn.functional.log_softmax: Op.log_softmax,
    torch.nn.functional.softmax: Op.softmax,
    torch.softmax: Op.softmax,
    torch.nn.functional.layer_norm: Op.norm,
    torch.nn.functional.embedding: Op.embedding,
    torch.nn.functional.linear: Op.linear,
    # operator functions
    operator.getitem: Op.tuple_getitem,
    operator.setitem: Op.setitem,
    operator.add: Op.add,
    operator.iadd: Op.inplace_add,
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
    operator.or_: Op.bitwise_or_tensor,
    operator.invert: Op.bitwise_not,
    operator.mod: Op.remainder_tensor_tensor,
    operator.floordiv: Op.div_mod,
    # tensor methods (as strings)
    "size": Op.size,
    "add": Op.add,
    "add_": Op.inplace_add,
    "sub": Op.sub,
    "mul": Op.mul,
    "div": Op.div,
    "eq": Op.eq,
    "__eq__": Op.eq,
    "ne": Op.ne,
    "__ne__": Op.ne,
    "lt": Op.lt,
    "__lt__": Op.lt,
    "le": Op.le,
    "__le__": Op.le,
    "gt": Op.gt,
    "__gt__": Op.gt,
    "ge": Op.ge,
    "__ge__": Op.ge,
    "relu": Op.relu,
    "to": Op.cast,
    "sigmoid": Op.sigmoid,
    "reshape": Op.view,
    "repeat": Op.repeat,
    "cat": Op.cat,
    "index_put_": Op.index_put,
    "stack": Op.stack,
    "clone": Op.clone,
    "contiguous": Op.contiguous,
    "t": Op.permute_view,
    "permute": Op.permute_view,
    "transpose": Op.permute_view,
    "movedim": Op.permute_view,
    "squeeze": Op.squeeze_view,
    "unsqueeze": Op.unsqueeze_view,
    "narrow": Op.narrow_view,
    "select": Op.select_view,
    "unbind": Op.unbind_view,
    "neg": Op.neg,
    "square": Op.square,
    "rsqrt": Op.rsqrt,
    "view": Op.view,  # view is often used like reshape
    "expand": Op.expand,
    "copy_": Op.inplace_copy,
    "index_copy_": Op.inplace_index_copy,
    "masked_fill_": Op.inplace_masked_fill_tensor,
    "masked_fill": Op.masked_fill_tensor,
    "softmax": Op.softmax,
    "log_softmax": Op.log_softmax,
    "topk": Op.topk,
    "where": Op.where,
    "var_mean": Op.var_mean,
    "fill_": Op.inplace_fill_tensor,
    "index_select": Op.index_select,
    "bitwise_or": Op.bitwise_or_tensor,
    "__or__": Op.bitwise_or_tensor,
    # dtype cast-like tensor methods
    "long": Op.cast,
    "float": Op.cast,
    "int": Op.cast,
    "split": Op.split_with_size_view,
    "chunk": Op.split_with_size_view,
    "flatten": Op.flatten_view,
    "sum": Op.reduce_sum,
    "amax": Op.amax,
    "mean": Op.reduce_mean,
    "pow": Op.pow_scalar,
    "argsort": Op.argsort,
    "new_empty": Op.new_empty,
    "tril": Op.tril,
}

if TORCH_NPU_INSTALLED:
    _NPU_OP_MAP = {
        # torch.ops.npu functions
        torch.ops.npu.npu_moe_init_routing_v2: Op.moe_init_routing_v3,
        torch.ops.npu.npu_moe_re_routing: Op.moe_re_routing,
        torch.ops.npu.npu_add_rms_norm: Op.add_rms_norm,
        torch.ops.npu.npu_rms_norm: Op.rms_norm,
        torch.ops.npu.npu_scatter_nd_update: Op.scatter_nd_update,
        torch.ops.npu.npu_scatter_nd_update_: Op.scatter_nd_update_,
        torch.ops.npu.npu_moe_token_unpermute: Op.moe_token_unpermute,
        torch.ops.npu.npu_swiglu: Op.swiglu,
        torch.ops.npu.npu_moe_gating_top_k: Op.moe_gating_top_k,
        torch.ops.npu.npu_moe_gating_top_k_softmax: Op.moe_gating_top_k_softmax,
        torch.ops.npu.npu_moe_distribute_combine_v2: Op.moe_distribute_combine_v2,
        torch.ops.npu.npu_moe_distribute_dispatch_v2: Op.moe_distribute_dispatch_v2,
        torch.ops.npu.npu_apply_rotary_pos_emb: Op.apply_rotary_pos_emb,
        torch.ops.npu.npu_grouped_matmul: Op.grouped_matmul,
        torch.ops.npu.npu_fused_infer_attention_score: Op.fused_infer_attention_score,
        torch.ops.npu.npu_add_rms_norm_quant: Op.add_rms_norm_quant,
        torch.ops.npu.npu_dequant_swiglu_quant: Op.dequant_swiglu_quant,
        torch.ops.npu.npu_quantize: Op.npu_quantize,
        torch.ops.npu.npu_quant_matmul: Op.quant_matmul,
        torch.ops.npu.npu_dynamic_quant: Op.npu_dynamic_quant,
        torch.ops.npu.npu_interleave_rope: Op.interleave_rope,
    }
    _OP_MAP.update(_NPU_OP_MAP)

    _ATB_OP_MAP = {}


    def _register_atb_op(name, op_enum):
        atb_op = getattr(torch.ops.atb, name, None)
        if atb_op is None:
            return
        _ATB_OP_MAP[atb_op] = op_enum
        overload = getattr(atb_op, "default", None)
        if overload is not None:
            _ATB_OP_MAP[overload] = op_enum


    _register_atb_op("_npu_paged_attention", Op.paged_attention)
    _register_atb_op("_npu_reshape_and_cache", Op.reshape_and_cache)
    _OP_MAP.update(_ATB_OP_MAP)


def _convert_operator_to_torch_op(op):
    """Convert python operator to torch operator."""
    operator_map = {
        torch.nn.functional.layer_norm: torch.layer_norm,
        torch.nn.functional.log_softmax: torch.log_softmax,
        operator.add: torch.add,
        operator.iadd: "add_",
        operator.sub: torch.sub,
        operator.mul: torch.mul,
        operator.truediv: torch.div,
        operator.eq: torch.eq,
        operator.ne: torch.ne,
        operator.lt: torch.lt,
        operator.le: torch.le,
        operator.gt: torch.gt,
        operator.ge: torch.ge,
        operator.matmul: torch.matmul,
        operator.neg: torch.neg,
        operator.and_: torch.bitwise_and,
        operator.or_: torch.bitwise_or,
        operator.invert: torch.bitwise_not,
        operator.mod: torch.remainder,
        operator.floordiv: torch.floor_divide,
    }
    if op in operator_map:
        return operator_map[op]
    return op


_OP_MATCHERS = [
    ("call_function", "_log_api_usage_once"),
]

_DISABLE_VIEW_OPS_ENV = "MS_INFERRT_DISABLE_VIEW_OPS"

_VIEW_OP_SWITCH_NAMES = {
    Op.view: frozenset(("view", "reshape")),
    Op.permute_view: frozenset(("permute_view",)),
    Op.flatten_view: frozenset(("flatten",)),
    Op.slice_view: frozenset(("slice",)),
    Op.select_view: frozenset(("select",)),
    Op.squeeze_view: frozenset(("squeeze",)),
    Op.unsqueeze_view: frozenset(("unsqueeze",)),
    Op.narrow_view: frozenset(("narrow",)),
    Op.unbind_view: frozenset(("unbind",)),
    Op.split_with_size_view: frozenset(("split", "split_with_size")),
    Op.split_tensor_view: frozenset(("split", "split_tensor")),
    torch.transpose: frozenset(("transpose",)),
    aten.transpose.int: frozenset(("transpose",)),
    "transpose": frozenset(("transpose",)),
    torch.movedim: frozenset(("movedim",)),
    aten.movedim.int: frozenset(("movedim",)),
    aten.movedim.intlist: frozenset(("movedim",)),
    "movedim": frozenset(("movedim",)),
    torch.t: frozenset(("t",)),
    aten.t.default: frozenset(("t",)),
    "t": frozenset(("t",)),
    torch.permute: frozenset(("permute",)),
    aten.permute.default: frozenset(("permute",)),
    "permute": frozenset(("permute",)),
    torch.chunk: frozenset(("chunk",)),
    aten.chunk.default: frozenset(("chunk",)),
    "chunk": frozenset(("chunk",)),
}

# Disabling view ops only applies to entries with a registered non-view fallback.
_VIEW_OP_FALLBACKS = {
    Op.permute_view: Op.permute,
    Op.split_with_size_view: Op.split_with_size,
    Op.split_tensor_view: Op.split_tensor,
}


def _parse_disabled_view_ops():
    """Return normalized view-op switch tokens from MS_INFERRT_DISABLE_VIEW_OPS."""
    raw_value = os.environ.get(_DISABLE_VIEW_OPS_ENV, "")
    return {token.strip().lower() for token in raw_value.split(",") if token.strip()}


def _is_chunk_target(target):
    return target in (torch.chunk, aten.chunk.default) or (isinstance(target, str) and target == "chunk")


def _get_view_op_names(op, target):
    """Return switch names matching the selected InferRT view op and FX target."""
    if op == Op.permute_view:
        return _VIEW_OP_SWITCH_NAMES.get(target, _VIEW_OP_SWITCH_NAMES[Op.permute_view])

    if _is_chunk_target(target) and op in (Op.split_with_size_view, Op.split_tensor_view):
        return _VIEW_OP_SWITCH_NAMES.get(op, frozenset()) | _VIEW_OP_SWITCH_NAMES[target]
    return _VIEW_OP_SWITCH_NAMES.get(op, frozenset())


def _get_view_op_fallback(op):
    """Return the non-view fallback op where one exists."""
    return _VIEW_OP_FALLBACKS.get(op)


def _is_view_op(op):
    return bool(_get_view_op_names(op, None))


def _maybe_disable_view_op(op, target):
    """Return fallback op when the requested view op is disabled by environment."""
    disabled_view_ops = _parse_disabled_view_ops()
    if not disabled_view_ops:
        return op

    view_op_names = _get_view_op_names(op, target)
    if view_op_names and ("all" in disabled_view_ops or view_op_names & disabled_view_ops):
        fallback_op = _get_view_op_fallback(op)
        if fallback_op is None:
            print(f"Disabling InferRT view op {op} for target {target} is ignored: "
                  "no non-view implementation is registered, continue using view implementation.")
            return op
        return fallback_op
    return op


def _get_op(target):
    """Get the corresponding Op enum for a given target."""
    if isinstance(target, str):
        op = _OP_MAP.get(target)
        if op is not None:
            return op
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
            if node_module.startswith("torch._ops.ms_inferrt_dvm"):
                return Op.dvm_call
            if node_module.startswith("torch._ops.vllm"):
                return Op.python_call

    return Op.custom_call


def _check_and_fallback_op_by_backend_support(
        op: Op, output_value: Any, input_nodes: List[Any]
) -> Op:
    """
    Check whether the given op is supported by the target backend; if not, fall back to Op.custom_call.

    Args:
        op: The op enum to check.
        output_value: The output value (shape/dtype) for the op.
        input_nodes: List of input nodes.

    Returns:
        The same op if supported, or Op.custom_call when unsupported or on check failure.
    """
    if op in (Op.custom_call, Op.python_call, Op.dvm_call, Op.make_tuple):
        return op
    if not hasattr(op, "name"):
        return op

    try:
        input_values = [n.output for n in input_nodes]
        status, msg = _ms_inferrt_ir.check_op_support(op.name, output_value, input_values)
        if int(status) != 0:
            print(f"Op {op.name} not supported: {msg}, fallback to custom_call")
            return Op.custom_call
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Failed to check op support: {e}")
        return op

    return op


def _is_shape_sequence(arg):
    """
    Determines whether the given argument represents shape information,
    including direct sequence types or torch.fx.Node with shape-like example_value.

    Args:
        arg: The argument to check

    Returns:
        bool: True if the argument represents a shape sequence, False otherwise
    """
    if isinstance(arg, (tuple, list, torch.Size, immutable_list)):
        return True
    if isinstance(arg, torch.fx.Node):
        example_value = arg.meta.get("example_value", None)
        return isinstance(example_value, (tuple, list, torch.Size))
    return False


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
        if isinstance(value, torch.fx.node.Node):
            return value
        if isinstance(value, (list, tuple)):
            return value
        if value is None:
            return value
        if not arg_len:
            return [value]
        return [value for _ in range(arg_len)]

    # Handle Device string conversion
    type_name = str(type(value_type).__name__) + str(value_type)
    if "Device" in type_name and isinstance(value, str):
        return torch.device(value)

    return value


def _get_example_value_if_node(value: Any) -> Any:
    """
    Helper to get runtime example_value from a torch.fx.Node if available.
    Otherwise, return the value itself.
    """
    if isinstance(value, torch.fx.Node):
        return value.meta.get("example_value", None)
    return value


class _SymTypeInfo(NamedTuple):
    schema_int_types: Tuple[Any, ...]
    schema_float_types: Tuple[Any, ...]
    schema_bool_types: Tuple[Any, ...]
    sym_int_vals: Tuple[Any, ...]
    sym_float_vals: Tuple[Any, ...]
    sym_bool_vals: Tuple[Any, ...]


def _collect_sym_type_info() -> _SymTypeInfo:
    """Collect symbolic schema/runtime types if available on this torch version."""
    symint_type = getattr(torch, "SymIntType", None)
    symfloat_type = getattr(torch, "SymFloatType", None)
    symbool_type = getattr(torch, "SymBoolType", None)

    sym_int_cls = getattr(torch, "SymInt", None)
    sym_float_cls = getattr(torch, "SymFloat", None)
    sym_bool_cls = getattr(torch, "SymBool", None)

    schema_int_types: Tuple[Any, ...] = (torch.IntType,)
    if symint_type is not None:
        schema_int_types = schema_int_types + (symint_type,)

    schema_float_types: Tuple[Any, ...] = (torch.FloatType,)
    if symfloat_type is not None:
        schema_float_types = schema_float_types + (symfloat_type,)

    schema_bool_types: Tuple[Any, ...] = (torch.BoolType,)
    if symbool_type is not None:
        schema_bool_types = schema_bool_types + (symbool_type,)

    sym_int_vals = tuple(t for t in (sym_int_cls,) if t is not None)
    sym_float_vals = tuple(t for t in (sym_float_cls,) if t is not None)
    sym_bool_vals = tuple(t for t in (sym_bool_cls,) if t is not None)

    return _SymTypeInfo(
        schema_int_types=schema_int_types,
        schema_float_types=schema_float_types,
        schema_bool_types=schema_bool_types,
        sym_int_vals=sym_int_vals,
        sym_float_vals=sym_float_vals,
        sym_bool_vals=sym_bool_vals,
    )


def _check_runtime_value_against_type(value_type, runtime_v: Any) -> bool:
    """
    Check concrete runtime value against schema type. Caller ensures runtime_v is not None.

    To avoid over-constraining schema resolution, we explicitly check common primitive
    types and return True for unrecognized schema types.
    """
    info = _collect_sym_type_info()

    # Standard ScalarType (e.g. dtype indicators)
    if "ScalarType" in str(value_type):
        return isinstance(runtime_v, (torch.dtype, int))

    # Tensor types
    if isinstance(value_type, torch.TensorType):
        return isinstance(runtime_v, torch.Tensor)

    # Int / SymInt
    if isinstance(value_type, info.schema_int_types):
        # Allow Python int, symbolic int, and torch.dtype.
        return isinstance(runtime_v, (int, torch.dtype) + info.sym_int_vals)

    # Float / SymFloat
    if isinstance(value_type, info.schema_float_types):
        # Allow float / int / SymInt / SymFloat
        return isinstance(
            runtime_v, (float, int) + info.sym_int_vals + info.sym_float_vals
        )

    # Number (more generic numeric type)
    if isinstance(value_type, torch.NumberType):
        # Allow all number-like: int/float/bool + Sym*
        return isinstance(
            runtime_v,
            (int, float, bool)
            + info.sym_int_vals
            + info.sym_float_vals
            + info.sym_bool_vals,
        )

    # Bool / SymBool
    if isinstance(value_type, info.schema_bool_types):
        return isinstance(runtime_v, (bool,) + info.sym_bool_vals)

    # String
    if isinstance(value_type, torch.StringType):
        return isinstance(runtime_v, str)

    # Device / Layout / MemoryFormat like enum types
    type_name = str(type(value_type).__name__) + str(value_type)
    if "Device" in type_name:
        return isinstance(runtime_v, (torch.device, str))
    if "Layout" in type_name:
        return isinstance(runtime_v, torch.layout)
    if "MemoryFormat" in type_name:
        return isinstance(runtime_v, torch.memory_format)

    # For other unknown types: to avoid incorrectly filtering schemas, we treat them as compatible
    # and let higher-level logic further disambiguate if needed.
    return True


def _is_value_compatible_with_type(value_type, value: Any) -> bool:
    """
    Check whether a Python value (or FX Node) is compatible with a torch schema value_type.

    This is used to disambiguate between multiple overload schemas, so we keep it intentionally
    conservative: if we cannot confidently decide, we return True to avoid false negatives.
    """
    if isinstance(value_type, torch.OptionalType):
        if value is None:
            return True
        return _is_value_compatible_with_type(value_type.getElementType(), value)

    if isinstance(value_type, torch.ListType):
        elem_type = value_type.getElementType()
        if isinstance(value, torch.fx.Node):
            example_value = value.meta.get("example_value", None)
            if example_value is None:
                return True
            if isinstance(example_value, (list, tuple)):
                return all(_is_value_compatible_with_type(elem_type, v) for v in example_value)
            return _is_value_compatible_with_type(elem_type, example_value)
        if isinstance(value, (list, tuple)):
            if not value:
                return True
            return all(_is_value_compatible_with_type(elem_type, v) for v in value)
        return _is_value_compatible_with_type(elem_type, value)

    runtime_v = _get_example_value_if_node(value)
    if runtime_v is None:
        return True
    return _check_runtime_value_against_type(value_type, runtime_v)


def _all_explicit_fx_inputs_are_scalars(args: Any, kwargs: Dict[str, Any]) -> bool:
    """
    True only when every positional argument and every kwarg value is a scalar runtime value
    (Python int/float/bool or symbolic Sym*). Used to skip per-argument schema compatibility
    checks in _create_args when the whole call is scalar-only.
    """
    info = _collect_sym_type_info()
    scalar_types = (int, float, bool) + info.sym_int_vals + info.sym_float_vals + info.sym_bool_vals

    for a in args:
        if isinstance(a, Node):
            ev = a.meta.get("example_value", None)
            if ev is None or not isinstance(ev, scalar_types):
                return False
        elif not isinstance(a, scalar_types):
            return False

    for v in kwargs.values():
        if isinstance(v, Node):
            ev = v.meta.get("example_value", None)
            if ev is None or not isinstance(ev, scalar_types):
                return False
        elif not isinstance(v, scalar_types):
            return False

    return True


def _create_args(schema: torch.FunctionSchema, node: Node, custom_args=None) -> List[Argument]:
    """
    Create a list of Argument objects from a torch fx node.

    Args:
        schema (torch.FunctionSchema): The schema of the node.
        node (torch.fx.Node): The FX node whose arguments should be created.
        custom_args: Optional custom args to use instead of node.args.

    Returns:
        List[Argument]: A list of Argument objects in the node's arguments, preserving order.
        Bool: Whether the arguments are valid.
    """
    flat_args = []
    args = custom_args if custom_args is not None else node.args
    kwargs = node.kwargs
    arg_idx = 0

    # Special handling for view operation: PyTorch's view() accepts variable-length arguments,
    # allowing the shape to be specified as unpacked integers.
    if (node.target in ["view", "reshape", "repeat", "permute", "new_empty", "expand"]
            or node.target is torch.functional.einsum) \
            and not _is_shape_sequence(args[1]):
        args = [args[0], args[1:]]

    # Some factory ops (e.g. torch.empty) accept varargs size in Python,
    # while schema expects a single int[]/SymInt[] positional argument.
    # Normalize positional varargs into one shape sequence before schema matching.
    positional_schema_args = [a for a in schema.arguments if not a.kwarg_only]
    if (
        len(args) > 1
        and len(positional_schema_args) == 1
        and isinstance(positional_schema_args[0].real_type, torch.ListType)
    ):
        args = [list(args)]

    if len(args) + len(kwargs) > len(schema.arguments):
        return flat_args, False

    skip_type_compat_check = _all_explicit_fx_inputs_are_scalars(args, kwargs)

    for arg in args:
        if schema.arguments[arg_idx].kwarg_only:
            return flat_args, False

        # Additional type compatibility check to narrow down overloads.
        if not skip_type_compat_check and not _is_value_compatible_with_type(
                schema.arguments[arg_idx].real_type, arg
        ):
            return flat_args, False

        real_arg = _argument_to_real_value(
            schema.arguments[arg_idx].real_type, arg, schema.arguments[arg_idx].N
        )
        flat_args.append(real_arg)
        arg_idx += 1

    consumed_kwargs = 0
    for argument in schema.arguments[arg_idx:]:
        if argument.name in kwargs:
            kw_value = kwargs[argument.name]
            # Additional type compatibility check for kwargs.
            if not skip_type_compat_check and not _is_value_compatible_with_type(
                    argument.real_type, kw_value
            ):
                return flat_args, False

            real_arg = _argument_to_real_value(
                argument.real_type, kw_value, argument.N
            )
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
    aten_to_torch_map = {
        "aten::add": torch.add,
        "aten::sub": torch.sub,
        "aten::mul": torch.mul,
        "aten::div": torch.div,
        "aten::eq": torch.eq,
        "aten::ne": torch.ne,
        "aten::lt": torch.lt,
        "aten::le": torch.le,
        "aten::gt": torch.gt,
        "aten::ge": torch.ge,
    }

    if isinstance(target, OpOverload):
        if hasattr(target, "_schema") and hasattr(target._schema, "name"):
            schema_name = target._schema.name
            base_name = schema_name.split(".")[0]
            target = aten_to_torch_map.get(base_name, target)

    if isinstance(target, str):
        if not target.startswith("__"):
            for ns in iter(torch.ops):
                ops_ns = getattr(torch.ops, ns)
                if hasattr(ops_ns, target):
                    op_target = getattr(ops_ns, target)
                    if isinstance(op_target, (OpOverload, OpOverloadPacket)):
                        return (
                            op_target._qualified_op_name,
                            [getattr(op_target, overload)._schema for overload in op_target.overloads()],
                        )
        aten_fn = torch.jit._builtins._find_builtin(target)
        if aten_fn is not None:
            return aten_fn, torch._C._jit_get_schemas_for_operator(aten_fn)
        return None, None

    if isinstance(target, OpOverload):
        return target._schema.name, [target._schema]

    if isinstance(target, OpOverloadPacket):
        return (
            target._qualified_op_name,
            [getattr(target, overload)._schema for overload in target.overloads()],
        )

    aten_fn = torch.jit._builtins._find_builtin(target)
    if aten_fn is not None:
        return aten_fn, torch._C._jit_get_schemas_for_operator(aten_fn)

    return None, None


def _flatten_args(op: Op, node: Node) -> List[Argument]:
    """
    Flatten the arguments of a given FX node into a flat list of Argument objects.

    Args:
        op (Op): The ms_inferrt operation enumeration.
        node (Node): The FX node whose arguments should be flattened.

    Returns:
        List[Argument]: A flat list of all Argument objects in the node's arguments, preserving order.
    """
    flat_args = []
    torch_op = _convert_operator_to_torch_op(node.target)
    op_name, schemas = _get_op_schemas(torch_op)
    # For binary ops with all scalar/symbol inputs (e.g., add_scalar, div_mod_scalar),
    # return empty args list since these ops produce symbolic expressions
    # that do not require schema matching or runtime args.
    all_args = list(node.args) + list(node.kwargs.values())
    if not schemas:
        return None, all_args

    pre_flatten_hook = get_pre_flatten_hook(op) or get_pre_flatten_hook(node.target)
    custom_args = None
    if pre_flatten_hook is not None:
        custom_args = pre_flatten_hook(node)

    found = False
    for schema in schemas:
        flat_args, found = _create_args(schema, node, custom_args)
        if found:
            break
    if not found:
        schemas_str = "\n".join(f"  Schema[{idx}]: {schema}" for idx, schema in enumerate(schemas))
        err_msg = (
            f"Failed to find a valid schema for {node.target} with arguments {node.args} and kwargs {node.kwargs}. "
            f"All schemas tried:\n{schemas_str}"
        )
        raise ValueError(err_msg)
    return op_name, flat_args


def _map_args(
        args, env, executor: GraphExecutor, sym_mgr: SymbolicShapeManager
) -> List[Node]:
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

        value = sym_mgr.from_torch_with_sym(arg)
        return executor.add_value_node(value)

    return [_map_arg(arg) for arg in args]


def _handle_input_node(node, executor, sym_mgr, env):
    """Handle input node processing."""
    example_value = node.meta.get("example_value", None)
    output_value = sym_mgr.from_torch_with_sym(example_value)
    if isinstance(example_value, torch.nn.Parameter):
        env[node] = executor.add_parameter_node(output_value)
    else:
        env[node] = executor.add_input_node(output_value)


def _handle_input_nodes(input_nodes, executor, env, sym_mgr):
    """Handle input nodes processing."""
    non_symbol_input_nodes = []
    # handle sym int input nodes first to register symbols for later reference
    for node in input_nodes:
        if isinstance(node.meta.get("example_value"), torch.SymInt):
            _handle_input_node(node, executor, sym_mgr, env)
        else:
            non_symbol_input_nodes.append(node)
    # handle non sym int input nodes
    for node in non_symbol_input_nodes:
        _handle_input_node(node, executor, sym_mgr, env)


def _handle_get_attr_node(node, gm, executor, env):
    """Handle get_attr node processing."""
    target = node.target
    assert isinstance(target, str)

    attr_val = gm
    for part in target.split("."):
        attr_val = getattr(attr_val, part)

    env[node] = executor.add_value_node(from_torch(attr_val))


def _prepare_call_args(op, node, executor, env, sym_mgr):
    """Prepare arguments for call_function/call_method nodes."""
    op_name, flat_node_args = _flatten_args(op, node)

    if op == Op.python_call:
        module_name = node.target.__module__
        op_name = node.target.__name__
        flat_node_args = [module_name, op_name] + flat_node_args

    if op == Op.custom_call:
        if not is_op_registered_by_custom_or_torch(op_name):
            print(f"Unregistered custom/torch op: {op_name}, fallback to python_call")
            module_name = node.target.__module__
            op_name = node.target.__name__
            flat_node_args = [module_name, op_name] + flat_node_args
            op = Op.python_call
        else:
            op_name = op_name.replace("::", ".")
            flat_node_args = [op_name] + flat_node_args
    elif op == Op.dvm_call:
        op_name = node.target.__name__
        payload_json = get_dvm_payload(op_name)
        if payload_json is None:
            raise RuntimeError(
                f"Payload not registered for dvm op '{op_name}'. "
                f"Use register_dvm_op('{op_name}', payload_json) first."
            )
        flat_node_args = [payload_json] + flat_node_args
    elif op == Op.tuple_getitem:
        op, flat_node_args = getitem_process(node, flat_node_args)
    elif op == Op.setitem:
        op, flat_node_args = setitem_process(node, flat_node_args)
        if op == Op.python_call:
            module_name = node.target.__module__
            op_name = node.target.__name__
            flat_node_args = [module_name, op_name] + flat_node_args

    hook_func = get_arg_mapping_hook(op) or get_arg_mapping_hook(node.target)
    if hook_func is not None:
        flat_node_args = hook_func(node, flat_node_args, executor)

    return op, _map_args(flat_node_args, env, executor, sym_mgr)


def _try_handle_symbolic_only_op(node, executor, env, sym_mgr) -> bool:
    """
    Try to handle ops that only manipulate symbolic integers/shapes and have no runtime kernel.

    Returns:
        bool: True if the node was handled and `env[node]` was updated, False otherwise.
    """
    target = node.target

    # torch.sym_sum / torch.sym_min: produce a symbolic Value directly.
    target_name = getattr(target, "__name__", None)
    if (
        target_name in ("sym_sum", "sym_min")
        or target is getattr(torch, "sym_sum", None)
        or target is getattr(torch, "sym_min", None)
    ):
        example_value = node.meta.get("example_value", None)
        output_value = sym_mgr.from_torch_with_sym(example_value)
        env[node] = executor.add_value_node(output_value)
        return True

    return False


def _handle_call_node(node, executor, env, sym_mgr):
    """Handle call_function/call_method node processing."""
    op = _get_op(node.target)
    if op is None:
        raise NotImplementedError(f"Unsupported op: {node.target}")

    if _try_handle_symbolic_only_op(node, executor, env, sym_mgr):
        return

    ops_hook = get_ops_mapping_hook(op)
    if ops_hook is not None:
        _, flat_node_args = _flatten_args(op, node)
        op = ops_hook(op, node, flat_node_args, executor)
    op = _maybe_disable_view_op(op, node.target)

    op, input_nodes = _prepare_call_args(op, node, executor, env, sym_mgr)

    example_value = node.meta.get("example_value", None)
    output_value = sym_mgr.from_torch_with_sym(example_value)

    original_op = op
    op = _check_and_fallback_op_by_backend_support(op, output_value, input_nodes)
    if op != original_op:
        op, input_nodes = _prepare_call_args(op, node, executor, env, sym_mgr)

    # For handle fx graph node has different output type with inferrt node.
    output_hook = get_output_mapping_hook(op)
    if output_hook is not None:
        env[node] = output_hook(node, op, input_nodes, executor, sym_mgr)
        return

    env[node] = executor.add_op_node(op, input_nodes, output_value)


def _handle_output_node(node, executor, env, sym_mgr):
    """Handle output node processing."""
    input_nodes = _map_args(node.args, env, executor, sym_mgr)
    env[node] = input_nodes[0]
    executor.add_return_node(env[node])


def is_enable_dump_ir():
    """
    Return True if value of environment variable `MS_INFERRT_DEV_DUMP_IR` is `1`, otherwise False
    """
    return os.environ.get("MS_INFERRT_DEV_DUMP_IR", "") == "1"


def get_ir_file_name():
    """
    Get dump ir file name, format is `graph_rank{rank_id}_{pid}.txt` when enable distributed, otherwise the format
    is `graph_{pid}.txt`
    """
    if torch.distributed.is_initialized():
        return f"graph_rank{torch.distributed.get_rank()}_{os.getpid()}.txt"
    return f"graph_{os.getpid()}.txt"


def write_gm_graph(gm, graph_id, file_name):
    """
    Dump graph module to file
    """
    with open(file_name, "a+", encoding="utf-8") as f:
        f.write(f"======================fx graph {graph_id}======================\n")
        f.write(gm.print_readable(print_output=False))
        f.write("\n\n")
        f.write(str(gm.graph))
        f.write("\n\n\n")


def write_inferrt_graph(text, file_name):
    """
    Dump inferrt ir to file
    """
    with open(file_name, "a+", encoding="utf-8") as f:
        f.write(text)
        f.write("\n\n")


# pylint: disable=bad-continuation
# pylint: disable=unused-argument
def backend(gm: GraphModule, example_inputs: List[torch.Tensor]):
    """
    A torch.fx backend that converts a GraphModule to a da.runtime.GraphExecutor,
    and returns a callable that executes the compiled graph.
    """
    graph_id = _next_unique_graph_id()
    _remove_matched_nodes(gm, _OP_MATCHERS)
    if is_enable_dump_ir():
        write_gm_graph(gm, graph_id, get_ir_file_name())
    eliminate_redundant_copy_(gm)
    decompose_full_(gm)
    _decompose_ops_with_fake_mode(gm)
    _init_pre_flatten_hooks()
    _init_arg_mapping_hooks()
    _init_ops_mapping_hooks()
    _init_output_mapping_hooks()
    _init_ms_inferrt_config()

    executor = GraphExecutor(f"fx_graph_{graph_id}")
    sym_mgr = SymbolicShapeManager()
    env: Dict[Node, Any] = {}

    get_collective_info_from_torch(gm)
    set_device_context()

    with executor:
        fx_input_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
        _handle_input_nodes(fx_input_nodes, executor, env, sym_mgr)

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                pass
            elif node.op == "get_attr":
                _handle_get_attr_node(node, gm, executor, env)
            elif node.op in ("call_function", "call_method"):
                _handle_call_node(node, executor, env, sym_mgr)
            elif node.op == "call_module":
                raise NotImplementedError(
                    "call_module is not supported in this simple backend."
                )
            elif node.op == "output":
                _handle_output_node(node, executor, env, sym_mgr)
            else:
                raise NotImplementedError(f"Unsupported node op: {node.op}")

    _debug_print("Building Graph:")
    if is_enable_dump_ir():
        write_inferrt_graph(executor.dump_graph(print_stdout=False), get_ir_file_name())
    executor.build()

    ms_inferrt_input_nodes = [env[n] for n in fx_input_nodes]
    # AclGraph input staticization hints:
    # - graph_key: unique identity for per-graph cache of static tensors
    # - input_is_parameter: per-input flag so parameters are excluded from staticization
    # - non_parameter_tensor_indices: explicit indices of non-parameter tensor inputs
    graph_key = id(executor)
    input_example_values = [node.meta.get("example_value", None) for node in fx_input_nodes]
    input_is_parameter = [
        isinstance(v, torch.nn.Parameter) for v in input_example_values
    ]
    non_parameter_tensor_indices = [
        i
        for i, v in enumerate(input_example_values)
        if (not input_is_parameter[i]) and isinstance(v, torch.Tensor)
    ]

    def compiled_callable(*inputs: torch.Tensor):
        set_device_context()
        update_runtime_inputs(
            ms_inferrt_input_nodes,
            inputs,
            input_is_parameter,
            graph_key,
            non_parameter_tensor_indices,
        )
        result = executor.run()
        return to_torch(result)

    return compiled_callable
