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
from typing import List

from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, SequenceStage

from vllm_mindspore.collocation.collocator import CollocatorMaster

logger = init_logger("vllm.collocation.executor_base")


def wrapper_distributed_executor_base_init(fun):

    def new_init(*arg, **kwarg):
        fun(*arg, **kwarg)
        # collocate model
        self = arg[0]
        self.collocator = None
        enable_collocation = os.environ.get("ENABLE_COLLOCATION", False)
        if enable_collocation in ["true", "True", True]:
            self.collocator = CollocatorMaster()

    return new_init


def distributed_execute_model(
    self,
    execute_model_req: ExecuteModelRequest,
) -> List[SamplerOutput]:
    # TODO: unify into collective_rpc

    # Collocation collocator management
    is_prefill = None
    if self.collocator is not None:
        seq_group_metadata_list = execute_model_req.seq_group_metadata_list
        is_prefill = False
        for seq_group in seq_group_metadata_list:
            for seqData in seq_group.seq_data.values():
                if seqData.stage == SequenceStage.PREFILL:
                    is_prefill = True
                    break
            if is_prefill:
                break
        if is_prefill:
            self.collocator.wait_prefill_ready()
        else:
            self.collocator.wait_decode_ready()

    if self.parallel_worker_tasks is None:
        self.parallel_worker_tasks = self._run_workers(
            "start_worker_execution_loop",
            async_run_tensor_parallel_workers_only=True)

    # Only the driver worker returns the sampling results.
    driver_outputs = self._driver_execute_model(execute_model_req)
    assert driver_outputs is not None

    # Collocation collocator management

    if self.collocator is not None:
        if is_prefill:
            self.collocator.prefill_done()
        else:
            self.collocator.decode_done()
    return driver_outputs
