# SPDX-License-Identifier: Apache-2.0

# Functions are adapted from
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/v1/engine/core.py
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
from vllm.v1.engine.utils import EngineZmqAddresses
from vllm.v1.executor.abstract import Executor

from vllm_mindspore.config import stateless_destroy_socket_process_group


def shutdown(self):
    super(self.__class__, self).shutdown()
    if dp_group := getattr(self, "dp_group", None):
        # vllm-mindspore: Close vllm-mindspore's socket dp group.
        stateless_destroy_socket_process_group(dp_group)


def init_dp_engine_core_actor(
    self,
    vllm_config: VllmConfig,
    on_head_node: bool,
    addresses: EngineZmqAddresses,
    executor_class: type[Executor],
    log_stats: bool,
    dp_rank: int = 0,
    local_dp_rank: int = 0,
):
    self.addresses = addresses
    vllm_config.parallel_config.data_parallel_rank = dp_rank
    vllm_config.parallel_config.data_parallel_rank_local = \
        local_dp_rank

    # vllm-mindspore: Do not reset current_platform.device_control_env_var for
    # mp executor when data parallel use ray, because setting it after
    # `import mindspore` will error in set device.

    from vllm.v1.engine.core import DPEngineCoreActor

    super(DPEngineCoreActor, self).__init__(vllm_config, on_head_node, "",
                                            executor_class, log_stats)
