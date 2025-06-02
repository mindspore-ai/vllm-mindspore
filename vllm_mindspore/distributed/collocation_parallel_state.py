# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2024 The vLLM team.
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
# ============================================================================

import os
from typing import Optional, Union

from torch.distributed import Backend
from vllm.logger import init_logger

logger = init_logger("vllm.collocation.parallel_state")


def wrapper_group_coordinator_init(fun):

    def new_init(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        use_device_communicator: bool,
        use_message_queue_broadcaster: bool = False,
        group_name: Optional[str] = None,
    ):
        enabled_collocation = os.environ.get("ENABLE_COLLOCATION", False)
        if (enabled_collocation in ["True", "true", True]):
            logger.info("Multimodel collocation management is activated.")
            fun(self, group_ranks, local_rank, "gloo", use_device_communicator,
                use_message_queue_broadcaster, group_name)
        else:
            fun(self, group_ranks, local_rank, torch_distributed_backend,
                use_device_communicator, use_message_queue_broadcaster,
                group_name)

    return new_init
