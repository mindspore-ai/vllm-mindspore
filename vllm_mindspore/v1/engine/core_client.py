# SPDX-License-Identifier: Apache-2.0

# Functions are adapted from
# https://github.com/vllm-project/vllm/blob/v0.9.1vllm/v1/engine/core_client.py
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

from typing import Optional

from vllm.v1.engine import (EngineCoreOutputs, EngineCoreRequest,
                            EngineCoreRequestType)
from vllm.v1.engine.utils import CoreEngine


class MsCoreEngine(CoreEngine):
    """One per data parallel rank."""

    def __init__(self, index: int = 0, local: bool = True):
        super().__init__(index, local)
        self.num_reqs_in_flight = 0


def get_core_engine_for_request(self,
                                dp_rank: Optional[int] = None) -> CoreEngine:
    # vLLM-MindSpore Plugin: To make the request number between dp more
    # balanced, use 0.8.3 method instead. This may make the wave mechanism to
    # be ineffective. A more meaningful method based on request wave needs to
    # be proposed latter.
    return min(self.core_engines, key=lambda e: e.num_reqs_in_flight)


async def add_request_async(self, request: EngineCoreRequest) -> None:
    self._ensure_stats_update_task()

    request.current_wave = self.current_wave
    request.client_index = self.client_index

    chosen_engine = self.get_core_engine_for_request(
        request.data_parallel_rank)
    self.reqs_in_flight[request.request_id] = chosen_engine
    # vLLM-MindSpore Plugin: Add req num to the chosen engine, which will make
    # it not selected before reaching equilibrium.
    chosen_engine.num_reqs_in_flight += 1

    to_await = self._send_input(EngineCoreRequestType.ADD, request,
                                chosen_engine)
    if not self.engines_running:
        # Notify coordinator that we're sending a request
        await self.first_req_send_socket.send(chosen_engine.identity)

    await to_await

    self._ensure_output_queue_task()


async def process_engine_outputs(self, outputs: EngineCoreOutputs):
    if outputs.finished_requests and self.reqs_in_flight:
        for req_id in outputs.finished_requests:
            # vLLM-MindSpore Plugin: Reduce req num to the chosen engine, which
            # will make it more likely to be selected.
            if (engine := self.reqs_in_flight.pop(req_id, None)):
                engine.num_reqs_in_flight -= 1
