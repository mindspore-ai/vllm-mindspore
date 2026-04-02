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

"""Ascend aclgraph config"""

from ms_inferrt._aclgraph_config import AclGraphConf
import torch


__all__ = ['AclGraphConf']
acl_graph = AclGraphConf.Instance()
def begin_capture():
    _get_or_create_pool_id()
    acl_graph.begin_capture()

def end_capture():
    acl_graph.end_capture()

def _get_or_create_pool_id():
    if acl_graph.pool_id() == (-1, -1):
        cur_pool_id = torch.npu.graph_pool_handle()
        acl_graph.set_pool_id(cur_pool_id)

def set_op_capture_skip(op_capture_skip: list):
    acl_graph.set_op_capture_skip(op_capture_skip)
