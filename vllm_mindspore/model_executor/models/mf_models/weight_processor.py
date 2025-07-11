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
import numpy as np
from safetensors import safe_open
import mindspore as ms
from mindspore.communication.management import get_rank, get_group_size
from mindformers.version_control import is_310p
from mindformers.experimental.infer.core.utils import get_pp_world_size
from vllm_mindspore.utils import convert_np_to_ms_dtype
from vllm.distributed import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank

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
        self.is_310 = is_310p()
        self.pp_group_size = get_pp_world_size()
        # self.tp_group_size = get_group_size()
        self.tp_group_size = get_tensor_model_parallel_world_size()
        self.global_rank_id = get_rank()
        self.rank_id = get_tensor_model_parallel_rank()
        self.pp_stage = self.global_rank_id // self.tp_group_size
        self.parameter_dict = {}
        self.file_handles = {}
        self.is_split_param = self.tp_group_size > 1

    def get_layer_index(self, num_layers):
        offset = self.config.model.model_config.offset
        stage_layers = num_layers // self.pp_group_size
        start_layer_index = self.pp_stage * stage_layers
        end_layer_index = start_layer_index + stage_layers

        if self.pp_group_size > 1 and num_layers % self.pp_group_size != 0:
            for num in range(0, self.pp_stage):
                start_layer_index += offset[num]
                end_layer_index += offset[num]
            end_layer_index += offset[self.pp_stage]

        return start_layer_index, end_layer_index

    def get_file_handles(self, filename):
        if filename not in self.file_handles:
            fp = safe_open(filename, framework="np")
            self.file_handles[filename] = fp
        return self.file_handles[filename]

    def release_file_handles(self):
        del self.file_handles

    def get_safetensor_from_file(self, hf_param_name, src_hf_dir, hf_weight_map, is_split_param=False, split_axis=0):
        safetensor_file = hf_weight_map[hf_param_name]
        filename = os.path.join(src_hf_dir, safetensor_file)
        sf_file = self.get_file_handles(filename)
        qint4 = False
        if sf_file.metadata() is not None and hf_param_name in sf_file.metadata().keys():
            qint4 = True
        if not is_split_param:
            np_data = sf_file.get_tensor(hf_param_name)
            data_dtype = convert_np_to_ms_dtype(np_data)
            if self.is_310 and data_dtype == ms.bfloat16:
                np_data = np_data.astype(np.float32).astype(np.float16)
            return np_data, qint4

        np_data = sf_file.get_slice(hf_param_name)
        shape = np_data.get_shape()
        if split_axis == 0:
            split_size = shape[0] // self.tp_group_size
            start = self.rank_id * split_size
            stop = (self.rank_id + 1) * split_size
            split_data = np_data[start:stop]
        elif split_axis == 1:
            split_size = shape[1] // self.tp_group_size
            start = self.rank_id * split_size
            stop = (self.rank_id + 1) * split_size
            split_data = np_data[:, start:stop]
        elif split_axis == 2:
            split_size = shape[2] // self.tp_group_size
            start = self.rank_id * split_size
            stop = (self.rank_id + 1) * split_size
            split_data = np_data[:, :, start:stop]
        else:
            raise ValueError("split_axis:{} is not supported.".format(split_axis))
        data_dtype = convert_np_to_ms_dtype(split_data)
        if self.is_310 and data_dtype == ms.bfloat16:
            split_data = split_data.astype(np.float32).astype(np.float16)
        return split_data, qint4

    def split_weight_by_rank(self, weight, split_axis=0):
        shape = weight.shape
        if split_axis == 0:
            split_size = shape[0] // self.tp_group_size
            start = self.rank_id * split_size
            stop = (self.rank_id + 1) * split_size
            split_data = weight[start:stop]
        elif split_axis == 1:
            split_size = shape[1] // self.tp_group_size
            start = self.rank_id * split_size
            stop = (self.rank_id + 1) * split_size
            split_data = weight[:, start:stop]
        else:
            raise ValueError("split_axis:{} is not supported.".format(split_axis))
        return split_data

    def load_safetensors_shard(self, src_hf_dir):
        """ load safetensors and shards """
        raise NotImplementedError("load_safetensors_shard method is not implemented.")
