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
from safetensors import safe_open
from mindspore.communication.management import get_rank
from mindformers.experimental.infer.core.utils import get_tp_world_size
from mindformers.experimental.parallel_core.pynative.parallel_state import get_data_parallel_world_size

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
        self.tp_group_size = get_tp_world_size()
        self.dp_group_size = get_data_parallel_world_size()
        self.moe_ep_size = self.config.moe_config.moe_expert_parallel
        self.moe_tp_size = self.config.moe_config.moe_tensor_parallel
        self.tp_rank_id = self.global_rank_id % self.tp_group_size

        num_router_experts = self.config.moe_config.expert_num
        self.ep_group_nums = num_router_experts // self.moe_ep_size
        self.moe_ep_rank_id = self.global_rank_id // self.moe_tp_size
        self.moe_tp_rank_id = self.global_rank_id % self.moe_tp_size

        print(f"global_rank_id: {self.global_rank_id} \n"
              f"tp_group_size: {self.tp_group_size} \n"
              f"dp_group_size: {self.dp_group_size} \n"
              f"tp_rank_id: {self.tp_rank_id} \n"
              f"num_router_experts: {num_router_experts} \n"
              f"ep_group_nums: {self.ep_group_nums} \n"
              f"moe_ep_rank_id: {self.moe_ep_rank_id} \n"
              f"moe_tp_rank_id: {self.moe_tp_rank_id} \n"
              f"moe_ep_size: {self.moe_ep_size} \n"
              f"moe_tp_size: {self.moe_tp_size}", flush=True)

        self.parameter_dict = {}
        self.file_handles = {}

    def get_file_handles(self, filename):
        if filename not in self.file_handles:
            fp = safe_open(filename, framework="np")
            self.file_handles[filename] = fp
        return self.file_handles[filename]

    def release_file_handles(self):
        del self.file_handles

    def get_moe_safetensor_from_file(self, hf_param_name, src_hf_dir, hf_weight_map, is_split_param=False, split_axis=0):
        safetensor_file = hf_weight_map[hf_param_name]
        filename = os.path.join(src_hf_dir, safetensor_file)
        sf_file = self.get_file_handles(filename)
        qint4 = False
        if sf_file.metadata() is not None and hf_param_name in sf_file.metadata().keys():
            qint4 = True
        if not is_split_param or self.moe_tp_size == 1:
            np_data = sf_file.get_tensor(hf_param_name)
            return np_data, qint4

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

    def get_safetensor_from_file(self, hf_param_name, src_hf_dir, hf_weight_map, is_split_param=False, split_axis=0):
        safetensor_file = hf_weight_map[hf_param_name]
        filename = os.path.join(src_hf_dir, safetensor_file)
        sf_file = self.get_file_handles(filename)
        qint4 = False
        if sf_file.metadata() is not None and hf_param_name in sf_file.metadata().keys():
            qint4 = True
        if not is_split_param or self.tp_group_size == 1:
            np_data = sf_file.get_tensor(hf_param_name)
            return np_data, qint4

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
            start = self.rank_id * split_size
            stop = (self.rank_id + 1) * split_size
            split_data = np_data[:, :, start:stop]
        else:
            raise ValueError("split_axis:{} is not supported.".format(split_axis))
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
