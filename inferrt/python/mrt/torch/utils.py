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

'''
utils for converting between torch and mrt.ir.
'''
import os
from typing import Any

import torch
from torch import distributed as dist
from torch._C._distributed_c10d import _resolve_process_group

from mrt import _mrt_torch
from mrt.ir import Value, Tensor, Tuple
from mrt._mrt_collective import CollectiveManager

# pylint: disable=protected-access
_DIST_OP_LIST = [
    torch.ops._c10d_functional.all_gather_into_tensor,
    torch.ops._c10d_functional.all_reduce,
    torch.ops._c10d_functional.reduce_scatter_tensor,
    torch.ops._c10d_functional.all_to_all_single,
]

def _extract_global_comm_info():
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.getenv('LOCAL_RANK', "0"))
    world_size = dist.get_world_size()

    CollectiveManager.instance().set_global_rank_id(rank)
    CollectiveManager.instance().set_local_rank_id(local_rank)
    CollectiveManager.instance().set_global_rank_size(world_size)


def _set_communication_info(ptd):
    '''Get communication info from torch and set to CollectiveManager for a given process group.'''
    pg = _resolve_process_group(ptd)
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.getenv('LOCAL_RANK', "0"))
    world_size = dist.get_world_size()

    group_rank = dist.get_rank(pg)
    rank_list = dist.get_process_group_ranks(pg)

    hccl_comm_handle = pg._get_backend(torch.device('npu')).get_hccl_comm(rank)

    CollectiveManager.instance().set_global_rank_id(rank)
    CollectiveManager.instance().set_local_rank_id(local_rank)
    CollectiveManager.instance().set_global_rank_size(world_size)

    CollectiveManager.instance().create_communication_group(f"{ptd}", rank_list, group_rank, hccl_comm_handle)


def _extract_and_setup_comm_groups(node_args):
    ptd_arg = node_args[-1]
    if CollectiveManager.instance().is_group_exist(f"{ptd_arg}"):
        return
    _set_communication_info(ptd_arg)


def get_collective_info_from_torch(gm: torch.fx.GraphModule):
    '''
    Extract communication info from fx graph and set to CollectiveManager.
    '''
    if dist.is_initialized():
        _extract_global_comm_info()
        for node in gm.graph.nodes:
            if node.op in ("call_function", "call_method"):
                if node.target in _DIST_OP_LIST:
                    _extract_and_setup_comm_groups(node.args)


def from_torch(obj: Any) -> Value:
    '''
    Convert a torch object to mrt.ir.Value.
    '''
    if isinstance(obj, Value):
        return obj
    if isinstance(obj, torch.SymInt):
        return Value(-1)
    if isinstance(obj, (list, tuple)):
        return Value(Tuple([from_torch(e) for e in obj]))
    # pylint: disable=protected-access
    if isinstance(obj, torch._subclasses.FakeTensor):
        return Value(_mrt_torch.from_torch(obj, is_fake=True))
    if isinstance(obj, torch.Tensor):
        return Value(_mrt_torch.from_torch(obj))
    if isinstance(obj, (int, float, bool, str)):
        return Value(obj)
    if obj is None:
        return Value()
    raise TypeError(f"Unsupported python type for conversion to mrt.ir.Value: {type(obj)}")


def to_torch(value: Value) -> Any:
    '''
    Convert a mrt.ir.Value to torch object.
    '''
    if not isinstance(value, Value):
        return value
    if value.is_none():
        return None
    if value.is_tensor():
        return _mrt_torch.to_torch(value.to_tensor())
    if value.is_tuple():
        return tuple(to_torch(item) for item in value.to_tuple())
    if value.is_int():
        return value.to_int()
    if value.is_double():
        return value.to_double()
    if value.is_bool():
        return value.to_bool()
    if value.is_string():
        return value.to_string()
    raise TypeError(f"Unsupported mrt.ir.Value for conversion to python object: {value}")


def update_tensor_data(tensor: Tensor, torch_tensor: torch.Tensor):
    _mrt_torch.update_tensor_data(tensor, torch_tensor)

def set_device_context():
    _mrt_torch.set_device_context()
