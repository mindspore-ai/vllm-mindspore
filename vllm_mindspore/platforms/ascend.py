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
"""Ascend platform."""

from typing import TYPE_CHECKING, Optional

import torch
import os
import mindspore as ms

from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum, _Backend
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


class AscendPlatform(Platform):
    _enum = PlatformEnum.CUDA
    device_name: str = "cuda"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"

    @classmethod
    def get_default_attn_backend(cls, selected_backend: _Backend):
        """Get the default attention backend of a device."""
        return _Backend.FLASH_ATTN

    @classmethod
    def get_device_capability(
        cls,
        device_id: int = 0,
    ) -> Optional[DeviceCapability]:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get the name of a device."""
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get the total memory of a device in bytes."""
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        """
        Check if the current platform supports async output.
        """
        if enforce_eager:
            # from vllm.logger import init_logger
            # logger = init_logger(__name__)
            logger.warning(
                "To see benefits of async output processing, enable CUDA "
                "graph. Since, enforce-eager is enabled, async output "
                "processor cannot be used"
            )
            return False
        return True

    @classmethod
    def inference_mode(cls):
        """A device-specific wrapper of `torch.inference_mode`.

        This wrapper is recommended because some hardware backends such as TPU
        do not support `torch.inference_mode`. In such a case, they will fall
        back to `torch.no_grad` by overriding this method.
        """
        return torch.inference_mode(mode=True)

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        """
        Check and update the configuration for the current platform.

        It can raise an exception if the configuration is not compatible with
        the current platform, or it can update the configuration to make it
        compatible with the current platform.

        The config is passed by reference, so it can be modified in place.
        """
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config

        if parallel_config.worker_cls == "auto":
            import vllm.envs as envs

            if scheduler_config.is_multi_step:
                if envs.VLLM_USE_V1:
                    raise NotImplementedError
                else:
                    parallel_config.worker_cls = (
                        "vllm.worker.multi_step_worker.MultiStepWorker"
                    )
            elif vllm_config.speculative_config:
                if envs.VLLM_USE_V1:
                    raise NotImplementedError
                else:
                    parallel_config.worker_cls = (
                        "vllm.spec_decode.spec_decode_worker.create_spec_worker"
                    )
                    parallel_config.sd_worker_cls = "vllm.worker.worker.Worker"
            else:
                if envs.VLLM_USE_V1:
                    parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"
                else:
                    parallel_config.worker_cls = "vllm.worker.worker.Worker"

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        if os.getenv("ASCEND_TOTAL_MEMORY_GB"):
            total_device_memory = int(os.environ["ASCEND_TOTAL_MEMORY_GB"])
        else:
            total_device_memory = 64
            logger.warning(
                "Total device memory should be set by environ 'ASCEND_TOTAL_MEMORY_GB', "
                "please check size by cmd(npu-smi info). "
                "For now, we will try default size(64GB) which might not be correct exactly."
            )
        max_device_memory_for_ms = str(total_device_memory * cache_config.gpu_memory_utilization) + 'GB'
        ms.set_context(max_device_memory=max_device_memory_for_ms)
        logger.info("max_device_memory for mindspore is: ", max_device_memory_for_ms)

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        """
        Verify whether the quantization is supported by the current platform.
        """
        if cls.supported_quantization and quant not in cls.supported_quantization:
            raise ValueError(
                f"{quant} quantization is currently not supported in "
                f"{cls.device_name}."
            )
