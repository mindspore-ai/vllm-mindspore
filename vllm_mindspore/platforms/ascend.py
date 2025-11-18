# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/platforms/cuda.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2024-2025 The vLLM team.
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
"""Ascend platform."""

from typing import TYPE_CHECKING, Optional, Union

import torch
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms.interface import Platform, PlatformEnum

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
    dist_backend: str = "hccl"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        return None

    @classmethod
    def has_device_capability(
        cls,
        capability: Union[tuple[int, int], int],
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
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        torch.cuda.set_device(device)

    @classmethod
    def check_if_supports_dtype(cls, torch_dtype: torch.dtype):
        pass

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config

        if parallel_config.worker_cls == "auto":
            if vllm_config.speculative_config:
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
            # default value: 16 -> 128 for better performance
            cache_config.block_size = 128

        model_config = vllm_config.model_config
        model_config.disable_cascade_attn = True

        # Cache between p0 and p1 effective only one-on-one situations. In data
        # parallelelism, it is a one-to-many scenario, cache should be disabled.
        if (model_config.multimodal_config is not None
                and not model_config.disable_mm_preprocessor_cache
                and parallel_config.data_parallel_size > 1):
            model_config.multimodal_config.disable_mm_preprocessor_cache = True
            logger.info(
                "Disable mm preprocessor cache for data parallel size %d.",
                parallel_config.data_parallel_size)

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, head_size, dtype,
                             kv_cache_dtype, block_size, use_v1, use_mla,
                             has_sink, use_sparse):
        """Get the attention backend class of a device."""
        if use_mla:
            return "vllm_mindspore.v1.attention.backends.ms_attn.MLABackend"  # noqa E501
        return "vllm_mindspore.v1.attention.backends.ms_attn.MsAttentionBackend"  # noqa E501

    @classmethod
    def get_current_memory_usage(cls,
                                 device: Optional[torch.types.Device] = None
                                 ) -> float:
        """Return the memory usage in bytes."""
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.max_memory_allocated(device)

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """
        Get device specific communicator class for distributed communication.
        """
        if envs.VLLM_USE_V1:
            return "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # noqa E501
        return "vllm.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase"  # noqa E501

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

    @classmethod
    def use_all_gather(cls) -> bool:
        """
        Whether to use allgather in LogitsProcessor to gather the logits.
        """
        import vllm.envs as envs
        from vllm.config import get_current_vllm_config

        parallel_config = get_current_vllm_config().parallel_config
        return (envs.VLLM_USE_V1
                or parallel_config.distributed_executor_backend
                == "external_launcher")

    @classmethod
    def pre_register_and_update(cls, parser=None):
        if parser is not None:
            quant_action = parser._option_string_actions.get('--quantization')
            if quant_action and hasattr(quant_action,
                                        'choices') and quant_action.choices:
                ASCEND_QUANTIZATION_METHOD = ['ascend', 'golden-stick']
                if ASCEND_QUANTIZATION_METHOD not in quant_action.choices:
                    quant_action.choices.extend(ASCEND_QUANTIZATION_METHOD)
                    logger.debug("--quantization support ascend/golden-stick.")
