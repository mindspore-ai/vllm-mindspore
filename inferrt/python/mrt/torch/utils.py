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
utils for converting between torch and mrt.ir.
"""
from typing import Any, List, Tuple, Optional

import torch
from torch import distributed as dist
from torch._C._distributed_c10d import _resolve_process_group

from mrt import _mrt_torch
from mrt._mrt_api import is_custom_op_registered
from mrt.ir import Value, Tuple as MrtTuple, DataType, SymbolicVar
from mrt._mrt_collective import CollectiveManager


# pylint: disable=protected-access
_DIST_OP_LIST = [
    torch.ops._c10d_functional.all_gather_into_tensor,
    torch.ops._c10d_functional.all_reduce,
    torch.ops._c10d_functional.reduce_scatter_tensor,
    torch.ops._c10d_functional.all_to_all_single,
]


def _extract_global_comm_info():
    """Extract distributed communication information (rank, world_size)."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank > 7:
        raise ValueError(
            f"Expected rank id between 0 and 7, but received {rank}"
        )
    world_size = dist.get_world_size()

    CollectiveManager.instance().set_global_rank_id(rank)
    # TODO: Multi-machine scenario needs verification, current implementation only supports single machine with 8 NPUs
    CollectiveManager.instance().set_local_rank_id(rank)
    CollectiveManager.instance().set_global_rank_size(world_size)


def _set_communication_info(ptd):
    """Get communication info from torch and set to CollectiveManager for a given process group."""
    pg = _resolve_process_group(ptd)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size()

    group_rank = dist.get_rank(pg)
    rank_list = dist.get_process_group_ranks(pg)

    hccl_comm_handle = pg._get_backend(torch.device("npu")).get_hccl_comm(rank)

    CollectiveManager.instance().set_global_rank_id(rank)
    CollectiveManager.instance().set_local_rank_id(rank)
    CollectiveManager.instance().set_global_rank_size(world_size)

    CollectiveManager.instance().create_communication_group(
        f"{ptd}", rank_list, group_rank, hccl_comm_handle
    )


def _extract_and_setup_comm_groups(node_args):
    ptd_arg = node_args[-1]
    if CollectiveManager.instance().is_group_exist(f"{ptd_arg}"):
        return
    _set_communication_info(ptd_arg)


def get_collective_info_from_torch(gm: torch.fx.GraphModule):
    """
    Extract communication info from fx graph and set to CollectiveManager.
    """
    if dist.is_initialized():
        _extract_global_comm_info()
        for node in gm.graph.nodes:
            if node.op in ("call_function", "call_method"):
                if node.target in _DIST_OP_LIST:
                    _extract_and_setup_comm_groups(node.args)


def from_torch(obj: Any) -> Value:
    """
    Convert a torch object to mrt.ir.Value.
    """
    if isinstance(obj, Value):
        return obj
    if isinstance(obj, torch.SymInt):
        return Value(SymbolicVar(str(obj)))
    if isinstance(obj, (list, tuple)):
        return Value(MrtTuple([from_torch(e) for e in obj]))
    # pylint: disable=protected-access
    if isinstance(obj, torch._subclasses.FakeTensor):
        return Value(_mrt_torch.from_torch(obj, is_fake=True))
    if isinstance(obj, torch.Tensor):
        return Value(_mrt_torch.from_torch(obj))
    if isinstance(obj, (int, float, bool, str)):
        return Value(obj)
    if isinstance(obj, torch.device):
        device_str = str(obj).rsplit(".", maxsplit=1)[-1]
        return Value(device_str)
    if isinstance(obj, torch.dtype):
        dtype_str = str(obj).rsplit(".", maxsplit=1)[-1]  # "torch.float32" -> "float32"
        return Value(DataType.convert_str_to_int(dtype_str))
    if obj is None:
        return Value()
    raise TypeError(
        f"Unsupported python type for conversion to mrt.ir.Value: {type(obj)}"
    )


def to_torch(value: Value) -> Any:
    """
    Convert a mrt.ir.Value to torch object.
    """
    if not isinstance(value, Value):
        return value
    if value.is_none():
        return None
    if value.is_tensor():
        return _mrt_torch.to_torch(value.to_tensor())
    if value.is_tuple():
        return tuple(to_torch(item) for item in value.to_tuple())
    if value.is_int() or value.is_symbol():
        return value.to_int()
    if value.is_double():
        return value.to_double()
    if value.is_bool():
        return value.to_bool()
    if value.is_string():
        return value.to_string()
    raise TypeError(
        f"Unsupported mrt.ir.Value for conversion to python object: {value}"
    )


def set_device_context():
    _mrt_torch.set_device_context()


def _update_mrt_value(mrt_value: Value, torch_value: Any) -> Optional[Value]:
    """
    Update mrt.ir.Value with input value.
    Return None if the value is updated, otherwise return a new Value.
    """
    if mrt_value.is_tensor():
        _mrt_torch.update_tensor_data(mrt_value.to_tensor(), torch_value)
        return None
    if mrt_value.is_tuple():
        _update_tuple_data(mrt_value, torch_value)
        return None
    if mrt_value.is_symbol():
        mrt_symbol = mrt_value.to_symbol()
        if isinstance(mrt_symbol, SymbolicVar):
            mrt_symbol.set_value(int(torch_value))
        return None
    return from_torch(torch_value)


def _update_tuple_data(mrt_value: Value, torch_value: Any) -> None:
    """
    Update tuple data with input value.
    """
    mrt_tuple_items = mrt_value.to_tuple()
    if len(mrt_tuple_items) != len(torch_value):
        raise ValueError(
            f"Expected {len(mrt_tuple_items)} items in tuple, but received {len(torch_value)}"
        )
    for i, torch_item in enumerate(torch_value):
        mrt_item = mrt_tuple_items[i]
        new_value = _update_mrt_value(mrt_item, torch_item)
        if new_value is not None:
            mrt_tuple_items[i] = new_value


def update_runtime_inputs(
    param_nodes: List[Any],
    new_inputs: Tuple[Any, ...],
) -> None:
    """
    Update placeholder nodes with runtime input values and update symbolic variables.
    """
    if len(new_inputs) != len(param_nodes):
        raise ValueError(
            f"Expected {len(param_nodes)} inputs, but received {len(new_inputs)}"
        )

    for param_node, input_val in zip(param_nodes, new_inputs):
        new_value = _update_mrt_value(param_node.output, input_val)
        if new_value is not None:
            param_node.output = new_value


def tuple_indices_to_slice_arg(indices: Tuple[int, ...], shape: Tuple[int, ...]):
    """
    Convert tuple indices to slice arguments.
    """
    num_dims = len(shape)
    begin = []
    end = []
    axes = []
    steps = []
    processed_indices = []
    none_nums = 0
    # None in indices means expanding a dimension at the corresponding shape position, with size 1
    for idx in indices:
        if idx is None:
            none_nums += 1
    for idx in indices:
        if idx is Ellipsis:
            # Insert missing dimensions after ellipsis
            missing_dims = num_dims - len(indices) + 1 + none_nums
            processed_indices.extend(["ellipsis"] * missing_dims)
        else:
            processed_indices.append(idx)

    axis = 0
    for idx in processed_indices:
        if isinstance(idx, slice):
            begin.append(idx.start if idx.start is not None else 0)
            end.append(idx.stop if idx.stop is not None else shape[axis])
            steps.append(idx.step if idx.step is not None else 1)
            axes.append(axis)
        elif idx == "ellipsis":
            begin.append(0)
            end.append(shape[axis])
            steps.append(1)
            axes.append(axis)
        elif idx is None:
            continue
        else:
            begin.append(idx)
            end.append(idx + 1)
            steps.append(1)
            axes.append(axis)
        axis += 1
    return begin, end, steps, axes


def is_op_registered_by_custom_or_torch(full_op_name: str) -> bool:
    """
    Check if the full_op_name is registered in the custom operator registry or torch.ops registry.
    """
    if "." in full_op_name:
        op_namespace, op_name = full_op_name.rsplit(".", 1)
    elif "::" in full_op_name:
        op_namespace, op_name = full_op_name.rsplit("::", 1)
    else:
        op_namespace, op_name = None, full_op_name

    # Check if the op_name is registered in the custom operator registry.
    if is_custom_op_registered(op_name):
        return True

    # Check if the op_name is registered in the torch.ops registry.
    torch_ns = getattr(torch.ops, op_namespace)
    if hasattr(torch_ns, op_name):
        return True

    return False
