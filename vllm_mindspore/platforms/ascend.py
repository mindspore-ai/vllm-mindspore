#!/usr/bin/env python3
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
"""Ascend platform."""

from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms.interface import Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
else:
    ModelConfig = None
    VllmConfig = None

logger = init_logger(__name__)


class AscendPlatform(Platform):

    _enum = PlatformEnum.OOT
    device_name: str = "npu"
    device_type: str = "cuda"  # To use cuda worker, executor...
    simple_compile_backend: str = "npu"
    ray_device_key: str = "NPU"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        return True

    @classmethod
    def has_device_capability(
        cls,
        capability: Union[Tuple[int, int], int],
        device_id: int = 0,
    ) -> bool:
        return True

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get the name of a device."""
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def is_async_output_supported(cls, _) -> bool:
        """Check if the current platform supports async output."""
        return True

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        compilation_config = vllm_config.compilation_config
        model_config = vllm_config.model_config

        if parallel_config.worker_cls == "auto":
            if scheduler_config.is_multi_step:
                if envs.VLLM_USE_V1:
                    raise NotImplementedError(
                        "Multi-step scheduling is not supported (and not "
                        "needed) on vLLM V1. Please launch without "
                        "--num-scheduler-steps.")
                else:
                    parallel_config.worker_cls = \
                        "vllm.worker.multi_step_worker.MultiStepWorker"
            elif vllm_config.speculative_config:
                if envs.VLLM_USE_V1:
                    parallel_config.worker_cls = \
                            "vllm.v1.worker.gpu_worker.Worker"
                else:
                    parallel_config.worker_cls = \
                        "vllm.spec_decode.spec_decode_worker.create_spec_worker"
                    parallel_config.sd_worker_cls = \
                        "vllm.worker.worker.Worker"
            else:
                if envs.VLLM_USE_V1:
                    parallel_config.worker_cls = \
                            "vllm.v1.worker.gpu_worker.Worker"
                else:
                    parallel_config.worker_cls = "vllm.worker.worker.Worker"

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        model_config = vllm_config.model_config
        model_config.disable_cascade_attn = True

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, head_size, dtype,
                             kv_cache_dtype, block_size, use_v1, use_mla):
        """Get the attention backend class of a device."""
        if use_v1:
            if use_mla:
                return "vllm_mindspore.v1.attention.backends.ms_attn.MLABackend"
            return "vllm_mindspore.v1.attention.backends.ms_attn.MsAttentionBackend"
            raise RuntimeError("vLLM-MindSpore do not support v1 egine now!")
        if use_mla:
            logger.info("Using MindSpore MLA backend.")
            return "vllm_mindspore.attention.backends.ms_attn.MLABackend"

        if selected_backend == _Backend.FLASH_ATTN or selected_backend is None:
            logger.info("Using MindSpore Attention backend.")
            return "vllm_mindspore.attention.backends.ms_attn.MsAttentionBackend"

        raise ValueError(
            f"Invalid attention backend {str(selected_backend)} for vLLM-MindSpore with head_size: {str(head_size)}, dtype: {str(dtype)}, kv_cache_dtype: {str(kv_cache_dtype)}, block_size: {str(block_size)}."
        )

    @classmethod
    def get_current_memory_usage(cls,
                                 device: Optional[torch.types.Device] = None
                                 ) -> float:
        """Return the memory usage in bytes."""
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.max_memory_allocated(device)

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """Get device specific communicator class for distributed communication."""
        if envs.VLLM_USE_V1:
            return "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"
        return "vllm.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get the total memory of a device in bytes."""
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def supports_v1(cls, model_config: ModelConfig) -> bool:
        return True

    def get_punica_wrapper(cls) -> str:
        return "vllm_mindspore.lora.punica_wrapper.punica_npu.PunicaWrapperNPU"
