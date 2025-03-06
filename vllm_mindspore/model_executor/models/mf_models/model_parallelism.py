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
from safetensors import safe_open
from mindspore.communication.management import get_rank, get_group_size


class BaseModelParallelism:
    r"""
    Provide Infer model parameter convert and parallelism.
    Args:
        config (MF Config): The config of Infer model.
        network (InferenceModelForCausalLM): The network of infer model.

    """

    def __init__(self, config, network, is_quant):
        self.config = config
        self.network = network
        self.is_quant = is_quant

    def get_safetensor_from_file(self, hf_param_name, src_hf_dir, hf_weight_map, is_split_param=False, split_axis=0):
        tp_group_size = get_group_size()
        rank_id = get_rank()
        safetensor_file = hf_weight_map[hf_param_name]
        with safe_open(f"{src_hf_dir}/{safetensor_file}", framework="np") as sf_file:
            if not is_split_param:
                np_data = sf_file.get_tensor(hf_param_name)
                return np_data

            np_data = sf_file.get_slice(hf_param_name)
            shape = np_data.get_shape()
            if split_axis == 0:
                split_size = shape[0] // tp_group_size
                start = rank_id * split_size
                stop = (rank_id + 1) * split_size
                split_data = np_data[start:stop]
            elif split_axis == 1:
                split_size = shape[1] // tp_group_size
                start = rank_id * split_size
                stop = (rank_id + 1) * split_size
                split_data = np_data[:, start:stop]
            else:
                raise ValueError("split_axis:{} is not supported.".format(split_axis))
            return split_data

    def split_weight_by_rank(self, weight, split_axis=0):
        tp_group_size = get_group_size()
        rank_id = get_rank()
        shape = weight.shape
        if split_axis == 0:
            split_size = shape[0] // tp_group_size
            start = rank_id * split_size
            stop = (rank_id + 1) * split_size
            split_data = weight[start:stop]
        elif split_axis == 1:
            split_size = shape[1] // tp_group_size
            start = rank_id * split_size
            stop = (rank_id + 1) * split_size
            split_data = weight[:, start:stop]
        else:
            raise ValueError("split_axis:{} is not supported.".format(split_axis))
        return split_data

    def infer_convert_and_parallelism(self, src_hf_dir):
        """ infer convert and parallelism """
        raise NotImplementedError("infer_convert_and_parallelism method is not implemented.")
