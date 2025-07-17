# SPDX-License-Identifier: Apache-2.0

# Functions are adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/v1/engine/core.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2025 The vLLM team.
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

from vllm.config import VllmConfig

from vllm_mindspore.config import stateless_destroy_socket_process_group

def _init_data_parallel(self, vllm_config: VllmConfig):
    dp_rank = vllm_config.parallel_config.data_parallel_rank
    dp_size = vllm_config.parallel_config.data_parallel_size
    local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local

    assert dp_size > 1
    assert 0 <= local_dp_rank <= dp_rank < dp_size

    self.dp_rank = dp_rank
    self.dp_group = vllm_config.parallel_config.stateless_init_dp_group()
    self.current_wave = 0

def shutdown(self):
    super(self.__class__, self).shutdown()
    if dp_group := getattr(self, "dp_group", None):
        stateless_destroy_socket_process_group(dp_group)
