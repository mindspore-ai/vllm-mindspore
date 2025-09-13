# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import mindspore as ms
from mindspore import mint

from vllm.config import VllmConfig
from vllm.model_executor.models.constant_size_cache import ConstantSizeCache

@dataclass
class Qwen3NextCacheParams:
    conv_state: ms.Tensor = None
    temporal_state: ms.Tensor = None
    state_indices_tensor: ms.Tensor = None

    def at_layer_idx(self, layer_idx: int):
        return Qwen3NextCacheParams(self.conv_state[layer_idx],
                                    self.temporal_state[layer_idx],
                                    self.state_indices_tensor)


class Qwen3NextCacheManager(ConstantSizeCache):
    def __init__(self, vllm_config: VllmConfig, num_layers: int):

        # Determine max batch size to set size of Qwen3NextCache
        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        super().__init__(max_batch_size)

        hf_config = vllm_config.model_config.hf_config
        num_heads = hf_config.linear_num_key_heads
        value_num_heads = hf_config.linear_num_value_heads
        key_head_dim = hf_config.linear_key_head_dim
        value_head_dim = hf_config.linear_value_head_dim
        hidden_size = num_heads * (key_head_dim * 2 +
                                   value_num_heads // num_heads * key_head_dim)
        conv_state_shape = (hf_config.linear_conv_kernel_dim,
                            hidden_size)
        conv_state = mint.empty(size=(num_layers, max_batch_size,
                                      *conv_state_shape),
                                dtype = vllm_config.model_config.dtype)

        temporal_state_shape = (value_num_heads,
                                key_head_dim,
                                value_head_dim)
        temporal_state = mint.empty(size=(num_layers, max_batch_size,
                                          *temporal_state_shape),
                                    dtype=ms.float32)

        self._qwen3_next_cache = (conv_state, temporal_state)

    @property
    def cache(self):
        return self._qwen3_next_cache

    def _copy_cache(self, from_index: int, to_index: int):
        for cache_t in self.cache:
            cache_t[:, to_index].copy_(cache_t[:, from_index],
                                       non_blocking=True)

    def current_run_tensors(self, **kwargs) -> Qwen3NextCacheParams:
        """
        Return the tensors for the current run's conv and ssm state.
        """
        request_ids_to_seq_ids = kwargs.get("request_ids_to_seq_ids", None)
        finished_requests_ids = kwargs.get("finished_requests_ids", None)
        if request_ids_to_seq_ids is None or finished_requests_ids is None:
            return Qwen3NextCacheParams()

        self._release_finished_requests(finished_requests_ids)
        state_indices = self._prepare_current_run_cache(
            request_ids_to_seq_ids, finished_requests_ids
        )
        state_indices_tensor = ms.Tensor(state_indices, dtype=ms.int32)

        return Qwen3NextCacheParams(*self.cache, state_indices_tensor)
