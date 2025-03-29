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
import torch

import vllm.envs as envs

from vllm.config import VllmConfig, CompilationConfig, CompilationLevel, logger
from vllm.utils import random_uuid
from vllm.logger import init_logger

logger = init_logger(__name__)


def _verify_quantization(self) -> None:
    # Donnot verify now.
    return


def vllm_config_post_init(self):
    """Verify configs are valid & consistent with each other."""
    if self.model_config is not None:
        self.model_config.verify_async_output_proc(self.parallel_config,
                                                   self.speculative_config,
                                                   self.device_config)
        self.model_config.verify_with_parallel_config(self.parallel_config)

    if self.cache_config is not None:
        self.cache_config.verify_with_parallel_config(self.parallel_config)

    if self.lora_config:
        self.lora_config.verify_with_cache_config(self.cache_config)
        self.lora_config.verify_with_model_config(self.model_config)
        self.lora_config.verify_with_scheduler_config(
            self.scheduler_config)
    if self.prompt_adapter_config:
        self.prompt_adapter_config.verify_with_model_config(
            self.model_config)

    if self.quant_config is None and \
        self.model_config is not None and self.load_config is not None:
        self.quant_config = VllmConfig._get_quantization_config(
            self.model_config, self.load_config)

    from vllm.platforms import current_platform
    if self.scheduler_config is not None and \
        self.model_config is not None and \
        self.scheduler_config.chunked_prefill_enabled and \
        self.model_config.dtype == torch.float32 and \
        current_platform.get_device_capability() == (7, 5):
        logger.warning_once(
            "Turing devices tensor cores do not support float32 matmul. "
            "To workaround this limitation, vLLM will set 'ieee' input "
            "precision for chunked prefill triton kernels.")

    if self.compilation_config is None:
        self.compilation_config = CompilationConfig()
    if envs.VLLM_USE_V1 and self.model_config is not None and \
        not self.model_config.enforce_eager:
        # NOTE(woosuk): Currently, we use inductor because the piecewise
        # CUDA graphs do not work properly with the custom CUDA kernels.
        # FIXME(woosuk): Disable inductor to reduce the compilation time
        # and avoid any potential issues with the inductor.
        self.compilation_config.custom_ops = ["none"]
        self.compilation_config.use_cudagraph = True
        self.compilation_config.use_inductor = True
        self.compilation_config.cudagraph_num_of_warmups = 1
        self.compilation_config.pass_config.enable_fusion = False
        self.compilation_config.pass_config.enable_reshape = False
        self.compilation_config.level = CompilationLevel.PIECEWISE

    self._set_cudagraph_sizes()

    if self.cache_config is not None and \
        self.cache_config.cpu_offload_gb > 0 and \
        self.compilation_config.level != CompilationLevel.NO_COMPILATION:
        logger.warning(
            "CPU offload is not supported with `torch.compile` yet."
            " Disabling `torch.compile`.")
        self.compilation_config.level = CompilationLevel.NO_COMPILATION

    if self.lora_config is not None and self.compilation_config.level !=\
            CompilationLevel.NO_COMPILATION:
        logger.warning("LoRA is not supported with `torch.compile` yet. "
                        "Disabling `torch.compile`.")
        self.compilation_config.level = CompilationLevel.NO_COMPILATION

    current_platform.check_and_update_config(self)

    if self.model_config and self.model_config.use_mla:
        logger.info("For MindSpore, MLA supports chunked prefill and prefix, "
                    "so keep them enable.")

    if not self.instance_id:
        self.instance_id = random_uuid()[:5]


def _verify_args(self) -> None:
    if (self.max_num_batched_tokens < self.max_model_len
            and not self.chunked_prefill_enabled):
        logger.warning(
            f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
            f"smaller than max_model_len ({self.max_model_len}). "
            "This effectively limits the maximum sequence length to "
            "max_num_batched_tokens and makes vLLM reject longer "
            "sequences. Please increase max_num_batched_tokens or "
            "decrease max_model_len.")

        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs}).")

        if self.num_lookahead_slots < 0:
            raise ValueError(
                "num_lookahead_slots "
                f"({self.num_lookahead_slots}) must be greater than or "
                "equal to 0.")

        if self.num_scheduler_steps < 1:
            raise ValueError(
                "num_scheduler_steps "
                f"({self.num_scheduler_steps}) must be greater than or "
                "equal to 1.")

        if self.max_num_partial_prefills < 1:
            raise ValueError(
                f"max_num_partial_prefills ({self.max_num_partial_prefills}) "
                "must be greater than or equal to 1.")
        elif self.max_num_partial_prefills > 1:
            if not self.chunked_prefill_enabled:
                raise ValueError("Chunked prefill must be enabled to set "
                                 "max_num_partial_prefills > 1.")

            if self.long_prefill_token_threshold > self.max_model_len:
                raise ValueError(
                    "long_prefill_token_threshold "
                    f"({self.long_prefill_token_threshold}) cannot be greater "
                    f"than the max_model_len ({self.max_model_len}).")

        if (self.max_long_partial_prefills
                < 1) or (self.max_long_partial_prefills
                         > self.max_num_partial_prefills):
            raise ValueError(
                f"max_long_partial_prefills ({self.max_long_partial_prefills}) "
                "must be greater than or equal to 1 and less than or equal to "
                f"max_num_partial_prefills ({self.max_num_partial_prefills}).")
