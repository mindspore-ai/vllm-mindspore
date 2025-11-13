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

import asyncio

import vllm.envs as envs
from mindspore.profiler import ProfilerActivity, ProfilerLevel
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tracing import init_tracer
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from vllm.transformers_utils.tokenizer import init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.processor import Processor
from vllm.v1.metrics.loggers import StatLoggerManager

from vllm_mindspore.v1.worker.profile import AdapterProfiler

logger = init_logger(__name__)


def AsyncLLM__init__(
    self,
    vllm_config,
    executor_class,
    log_stats: bool,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    mm_registry=MULTIMODAL_REGISTRY,
    use_cached_outputs: bool = False,
    log_requests: bool = True,
    start_engine_loop: bool = True,
    stat_loggers=None,
    client_addresses=None,
    client_count: int = 1,
    client_index: int = 0,
) -> None:
    if not envs.VLLM_USE_V1:
        raise ValueError(
            "Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False. "
            "This should not happen. As a workaround, try using "
            "AsyncLLMEngine.from_vllm_config(...) or explicitly set "
            "VLLM_USE_V1=0 or 1 and report this issue on Github.")

    # Ensure we can serialize custom transformer configs
    maybe_register_config_serialize_by_value()

    self.model_config = vllm_config.model_config
    self.vllm_config = vllm_config
    self.observability_config = vllm_config.observability_config
    self.log_requests = log_requests

    self.log_stats = log_stats or (stat_loggers is not None)
    if not log_stats and stat_loggers is not None:
        logger.info(
            "AsyncLLM created with log_stats=False and non-empty custom "
            "logger list; enabling logging without default stat loggers")

    if self.model_config.skip_tokenizer_init:
        self.tokenizer = None
    else:
        # Tokenizer (+ ensure liveness if running in another process).
        self.tokenizer = init_tokenizer_from_configs(
            model_config=vllm_config.model_config)

    # Processor (converts Inputs --> EngineCoreRequests).
    self.processor = Processor(
        vllm_config=vllm_config,
        tokenizer=self.tokenizer,
        mm_registry=mm_registry,
    )

    # OutputProcessor (converts EngineCoreOutputs --> RequestOutput).
    self.output_processor = OutputProcessor(self.tokenizer,
                                            log_stats=self.log_stats)
    if self.observability_config.otlp_traces_endpoint is not None:
        tracer = init_tracer("vllm.llm_engine",
                             self.observability_config.otlp_traces_endpoint)
        self.output_processor.tracer = tracer

    # EngineCore (starts the engine in background process).
    self.engine_core = EngineCoreClient.make_async_mp_client(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_stats=self.log_stats,
        client_addresses=client_addresses,
        client_count=client_count,
        client_index=client_index,
    )

    # Loggers.
    self.logger_manager = None
    if self.log_stats:
        self.logger_manager = StatLoggerManager(
            vllm_config=vllm_config,
            engine_idxs=self.engine_core.engine_ranks_managed,
            custom_stat_loggers=stat_loggers,
            enable_default_loggers=log_stats,
            client_count=client_count,
        )
        self.logger_manager.log_engine_initialized()

    self.output_handler = None
    try:
        # Start output handler eagerly if we are in the asyncio eventloop.
        asyncio.get_running_loop()
        self._run_output_handler()
    except RuntimeError:
        pass

    if envs.VLLM_TORCH_PROFILER_DIR:
        logger.info(
            "vllm-mindspore profiler enabled. AsyncLLM CPU traces will be collected under %s",  # noqa: E501
            envs.VLLM_TORCH_PROFILER_DIR)
        self.profiler = AdapterProfiler(envs.VLLM_TORCH_PROFILER_DIR, [
            ProfilerActivity.CPU,
        ],
                                        ProfilerLevel.LevelNone,
                                        mstx=True)
    else:
        self.profiler = None
