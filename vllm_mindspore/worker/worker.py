#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
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

"""Worker functions"""
import gc
import os
import math
from typing import Tuple, Optional

import torch

from vllm.config import VllmConfig
from vllm.distributed import (
    ensure_kv_transfer_initialized,
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
)

from vllm.logger import init_logger

from vllm_mindspore.utils import is_mindformers_model_backend
from vllm.model_executor import set_random_seed
from vllm.sequence import SequenceGroupMetadata
from vllm.sampling_params import SamplingParams


logger = init_logger(__name__)


def _prepare_input_for_warmup(model_config, model_runner, cache_engine, is_prefill):
    bs = 1
    seq_len = model_config.max_seq_len_to_capture if is_prefill else 1
    dummy_data = model_runner.input_registry.dummy_data_for_profiling(model_config, seq_len, model_runner.mm_registry)
    block_tables = [i for i in range(math.ceil(seq_len / cache_engine.block_size))]
    seqs = [
        SequenceGroupMetadata(
            request_id=str(idx),
            is_prompt=is_prefill,
            seq_data={idx: dummy_data.seq_data},
            sampling_params=SamplingParams(),
            block_tables={idx: block_tables},
            lora_request=None,
            multi_modal_data=None,
            multi_modal_placeholders=None,
        )
        for idx in range(bs)
    ]

    model_input = model_runner.prepare_model_input(seqs)
    block_tables = model_input.attn_metadata.block_tables
    if block_tables is not None and block_tables.numel() <= 0:
        model_input.attn_metadata.block_tables = torch.zeros((1, 1), dtype=torch.int32)
    return model_input


def _warm_up_model(self) -> None:
    # cache_engine is a list with length equal to the size of pipeline-parallel, and only pp=1 is supported.
    kv_cache = self.cache_engine[0].gpu_cache

    # warmup for decode
    if self.vllm_config.scheduler_config.is_multi_step:
        model_input = _prepare_input_for_warmup(self.model_config, self.model_runner._base_model_runner, self.cache_engine[0], False)
        self.model_runner._base_model_runner.execute_model(model_input, kv_cache, None)
    else:
        model_input = _prepare_input_for_warmup(self.model_config, self.model_runner, self.cache_engine[0], False)
        self.model_runner.execute_model(model_input, kv_cache, None)
    torch.cuda.synchronize()

    # Reset the seed to ensure that the random state is not affected by
    # the model initialization and profiling.
    set_random_seed(self.model_config.seed)
