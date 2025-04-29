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
# ============================================================================

"""
transform huggingface safetensor.
"""

import os
from enum import Enum
from safetensors import safe_open
from mindspore.communication.management import get_rank, get_group_size
from mindformers.experimental.infer.core.utils import get_tp_world_size, get_moe_tp_world_size, get_moe_ep_world_size
from mindformers.experimental.parallel_core.pynative.parallel_state import get_data_parallel_world_size


class EPMethod(Enum):
    """
    EP method enums
    """
    DEFAULT = 'default'
    ALLTOALL = 'alltoall'
    ALLGATHER = 'allgather'

class BaseWeightProcessor:
    r"""
    Provide model weight load and shards.
    Args:
        config (MF Config): The config of Infer model.
        network (InferenceModelForCausalLM): The network of infer model.

    """

    def __init__(self, config, network, is_quant):
        self.config = config
        self.network = network
        self.is_quant = is_quant
        self.global_rank_id = get_rank()
        self.global_group_size = get_group_size()
        self.tp_group_size = get_tp_world_size()
        self.dp_group_size = get_data_parallel_world_size()
        self.num_router_experts = self.config.moe_config.expert_num if self.config.moe_config.expert_num else 1
        self.moe_ep_size = self.config.moe_config.moe_expert_parallel if self.config.moe_config.moe_expert_parallel else 1
        self.moe_tp_size = self.config.moe_config.moe_tensor_parallel if self.config.moe_config.moe_tensor_parallel else 1
        self.ep_method = EPMethod.DEFAULT
        if self.dp_group_size > 1 and self.moe_ep_size == self.global_group_size:
            self.ep_method = EPMethod.ALLTOALL
        elif self.dp_group_size > 1:
            self.ep_method = EPMethod.ALLGATHER
        self.tp_rank_id = self.global_rank_id % self.tp_group_size

        self.ep_group_nums = self.num_router_experts // self.moe_ep_size
        self.moe_ep_rank_id = self.global_rank_id // self.moe_tp_size
        self.moe_tp_rank_id = self.global_rank_id % self.moe_tp_size
        self.ep_start = self.moe_ep_rank_id * self.ep_group_nums
        self.ep_stop = (self.moe_ep_rank_id + 1) * self.ep_group_nums

        self.parameter_dict = {}
        self.file_handles = {}

    def get_file_handles(self, filename):
        if filename not in self.file_handles:
            fp = safe_open(filename, framework="np")
            self.file_handles[filename] = fp
        return self.file_handles[filename]

    def release_file_handles(self):
        del self.file_handles

    def get_safetensor_from_file(self, hf_param_name, src_hf_dir, hf_weight_map):
        safetensor_file = hf_weight_map[hf_param_name]
        filename = os.path.join(src_hf_dir, safetensor_file)
        sf_file = self.get_file_handles(filename)
        qint4 = False
        if sf_file.metadata() is not None and hf_param_name in sf_file.metadata().keys():
            qint4 = True

        np_data = sf_file.get_tensor(hf_param_name)
        return np_data, qint4

    def get_safetensor_from_file_split_tp_group(self, hf_param_name, src_hf_dir, hf_weight_map, split_axis=0):
        safetensor_file = hf_weight_map[hf_param_name]
        filename = os.path.join(src_hf_dir, safetensor_file)
        sf_file = self.get_file_handles(filename)
        qint4 = False
        if sf_file.metadata() is not None and hf_param_name in sf_file.metadata().keys():
            qint4 = True

        np_data = sf_file.get_slice(hf_param_name)
        shape = np_data.get_shape()
        if split_axis == 0:
            split_size = shape[0] // self.tp_group_size
            start = self.tp_rank_id * split_size
            stop = (self.tp_rank_id + 1) * split_size
            split_data = np_data[start:stop]
        elif split_axis == 1:
            split_size = shape[1] // self.tp_group_size
            start = self.tp_rank_id * split_size
            stop = (self.tp_rank_id + 1) * split_size
            split_data = np_data[:, start:stop]
        elif split_axis == 2:
            split_size = shape[2] // self.tp_group_size
            start = self.tp_rank_id * split_size
            stop = (self.tp_rank_id + 1) * split_size
            split_data = np_data[:, :, start:stop]
        else:
            raise ValueError("split_axis:{} is not supported.".format(split_axis))
        return split_data, qint4

    def get_safetensor_from_file_split_global_group(self, hf_param_name, src_hf_dir, hf_weight_map, split_axis=0):
        safetensor_file = hf_weight_map[hf_param_name]
        filename = os.path.join(src_hf_dir, safetensor_file)
        sf_file = self.get_file_handles(filename)
        qint4 = False
        if sf_file.metadata() is not None and hf_param_name in sf_file.metadata().keys():
            qint4 = True

        np_data = sf_file.get_slice(hf_param_name)
        shape = np_data.get_shape()
        if split_axis == 0:
            split_size = shape[0] // self.global_group_size
            start = self.global_rank_id * split_size
            stop = (self.global_rank_id + 1) * split_size
            split_data = np_data[start:stop]
        elif split_axis == 1:
            split_size = shape[1] // self.global_group_size
            start = self.global_rank_id * split_size
            stop = (self.global_rank_id + 1) * split_size
            split_data = np_data[:, start:stop]
        elif split_axis == 2:
            split_size = shape[2] // self.global_group_size
            start = self.global_rank_id * split_size
            stop = (self.global_rank_id + 1) * split_size
            split_data = np_data[:, :, start:stop]
        else:
            raise ValueError("split_axis:{} is not supported.".format(split_axis))
        return split_data, qint4

    def get_safetensor_from_file_split_moe_tp_group(self, hf_param_name, src_hf_dir, hf_weight_map, split_axis=0):
        safetensor_file = hf_weight_map[hf_param_name]
        filename = os.path.join(src_hf_dir, safetensor_file)
        sf_file = self.get_file_handles(filename)
        qint4 = False
        if sf_file.metadata() is not None and hf_param_name in sf_file.metadata().keys():
            qint4 = True

        np_data = sf_file.get_slice(hf_param_name)
        shape = np_data.get_shape()
        if split_axis == 0:
            split_size = shape[0] // self.moe_tp_size
            start = self.moe_tp_rank_id * split_size
            stop = (self.moe_tp_rank_id + 1) * split_size
            split_data = np_data[start:stop]
        elif split_axis == 1:
            split_size = shape[1] // self.moe_tp_size
            start = self.moe_tp_rank_id * split_size
            stop = (self.moe_tp_rank_id + 1) * split_size
            split_data = np_data[:, start:stop]
        else:
            raise ValueError("split_axis:{} is not supported.".format(split_axis))
        return split_data, qint4

    def get_routed_safetensor_3_dim(self, hf_param_name, src_hf_dir, hf_weight_map, split_ep=False, split_tp=False,
                                    tp_axis=-1):
        '''get_routed_safetensor_3_dim'''
        safetensor_file = hf_weight_map[hf_param_name]
        filename = os.path.join(src_hf_dir, safetensor_file)
        sf_file = self.get_file_handles(filename)
        qint4 = False
        if sf_file.metadata() is not None and hf_param_name in sf_file.metadata().keys():
            qint4 = True
        if not split_tp and not split_ep:
            np_data = sf_file.get_tensor(hf_param_name)
            return np_data, qint4

        np_data = sf_file.get_slice(hf_param_name)
        if not split_tp and split_ep:
            split_data = np_data[self.ep_start:self.ep_stop, :, :]
            return split_data, qint4

        shape = np_data.get_shape()
        if tp_axis == 1:
            split_size = shape[1] // self.moe_tp_size
            start = self.moe_tp_rank_id * split_size
            stop = (self.moe_tp_rank_id + 1) * split_size
            split_data = np_data[self.ep_start:self.ep_stop, start:stop, :] if split_ep else np_data[:, start:stop, :]
        elif tp_axis == 2:
            split_size = shape[2] // self.moe_tp_size
            start = self.moe_tp_rank_id * split_size
            stop = (self.moe_tp_rank_id + 1) * split_size
            split_data = np_data[self.ep_start:self.ep_stop, :, start:stop] if split_ep else np_data[:, :, start:stop]
        else:
            raise ValueError("tp_axis:{} is not supported.".format(tp_axis))
        return split_data, qint4

    def get_routed_safetensor_2_dim(self, hf_param_name, src_hf_dir, hf_weight_map, split_ep=False, split_tp=False,
                                    tp_axis=-1):
        '''get_moe_routed_safetensor_2_dim'''
        safetensor_file = hf_weight_map[hf_param_name]
        filename = os.path.join(src_hf_dir, safetensor_file)
        sf_file = self.get_file_handles(filename)
        qint4 = False
        if sf_file.metadata() is not None and hf_param_name in sf_file.metadata().keys():
            qint4 = True
        if not split_tp and not split_ep:
            np_data = sf_file.get_tensor(hf_param_name)
            return np_data, qint4

        np_data = sf_file.get_slice(hf_param_name)
        if not split_tp and split_ep:
            split_data = np_data[self.ep_start:self.ep_stop, :]
            return split_data, qint4

        shape = np_data.get_shape()
        if tp_axis == 1:
            split_size = shape[1] // self.moe_tp_size
            start = self.moe_tp_rank_id * split_size
            stop = (self.moe_tp_rank_id + 1) * split_size
            split_data = np_data[self.ep_start:self.ep_stop, start:stop] if split_ep else np_data[:, start:stop]
        else:
            raise ValueError("split_tp is True but tp_axis:{} is not supported.".format(tp_axis))
        return split_data, qint4

    def split_weight_by_rank(self, weight, split_axis=0):
        if self.tp_group_size == 1:
            return weight

        shape = weight.shape
        if split_axis == 0:
            split_size = shape[0] // self.tp_group_size
            start = self.tp_rank_id * split_size
            stop = (self.tp_rank_id + 1) * split_size
            split_data = weight[start:stop]
        elif split_axis == 1:
            split_size = shape[1] // self.tp_group_size
            start = self.tp_rank_id * split_size
            stop = (self.tp_rank_id + 1) * split_size
            split_data = weight[:, start:stop]
        else:
            raise ValueError("split_axis:{} is not supported.".format(split_axis))
        return split_data

    def load_safetensors_shard(self, src_hf_dir):
        """ load safetensors and shards """
        raise NotImplementedError("load_safetensors_shard method is not implemented.")
