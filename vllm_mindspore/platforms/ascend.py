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
from vllm.config.compilation import CompilationLevel, CUDAGraphMode
from vllm.logger import init_logger
from vllm.platforms.interface import Platform, PlatformEnum

from vllm_mindspore.utils import is_310p

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
    def is_support_aclgraph(cls, vllm_config: VllmConfig) -> bool:
        if vllm_config.model_config is None:
            logger.warning_once("model config is None, not support aclgraph")
            return False

        if vllm_config.model_config.enforce_eager:
            logger.warning_once("enforce_eager is True, not support aclgraph")
            return False

        if is_310p():
            logger.warning_once(
                "current platform is 310p, not support aclgraph")
            return False

        if vllm_config.speculative_config is not None:
            logger.warning_once("current O3 is not support mtp")
            return False

        if ((vllm_config.parallel_config.data_parallel_size > 1)
                and ("expert_parallel" in vllm_config.additional_config) and
            (int(vllm_config.additional_config.get("expert_parallel", None))
             > 1)):
            # current dp + ep will get different graph input for
            # attn_padding_idx and ffn_paddind_idx shape, will
            # cause graph capture not hit the graph inputs shape
            logger.warning_once("dp and ep is not support")
            return False

        not_support_model_type = [
            "Glm4vForConditionalGeneration", "Glm4ForCausalLM",
            "MiniCPMForCausalLM"
        ]

        model_arch = vllm_config.model_config.architectures[0]
        logger.warning_once(f"model architectures {model_arch}"
                            "current not support aclgraph")
        if model_arch in not_support_model_type:
            logger.warning_once(f"model architectures {model_arch}"
                                " current not support aclgraph")
            return False

        return True

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            # default value: 16 -> 128 for better performance
            cache_config.block_size = 128

        model_config = vllm_config.model_config

        # Model config may not be initialized when calling `VllmConfig()`,
        # so check if `model_config` is `None`
        if model_config is not None:
            model_config.disable_cascade_attn = True

        if not cls.is_support_aclgraph(vllm_config=vllm_config):
            logger.warning_once("reset compilation level to no_compilation")
            vllm_config.compilation_config.level \
                                = CompilationLevel.NO_COMPILATION

        vllm_config.compilation_config.splitting_ops = list(
            vllm_config.compilation_config._attention_ops)
        if vllm_config.compilation_config.level == CompilationLevel.PIECEWISE:
            # aclgraph only support piecewise capture mode
            vllm_config.compilation_config.cudagraph_mode \
                                            = CUDAGraphMode.PIECEWISE
            vllm_config.compilation_config.cudagraph_num_of_warmups = 1

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
    def pre_register_and_update(cls, parser=None):
        if parser is not None:
            quant_action = parser._option_string_actions.get('--quantization')
            if quant_action and hasattr(quant_action,
                                        'choices') and quant_action.choices:
                ASCEND_QUANTIZATION_METHOD = ['ascend', 'golden-stick']
                if ASCEND_QUANTIZATION_METHOD not in quant_action.choices:
                    quant_action.choices.extend(ASCEND_QUANTIZATION_METHOD)
                    logger.debug("--quantization support ascend/golden-stick.")

    @classmethod
    def make_materializing_weight_loader(cls, original_weight_loader):
        """
        Ensure the parameter is materialized before loading weights, 
        as required by deferred initialization.
        """

        def _materialize_and_load(param, *args, **kwargs):
            param.init_data()
            out = original_weight_loader(param, *args, **kwargs)
            return out

        return _materialize_and_load

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        if is_310p():
            # bfloat16 is not supported on the 310p due to hardware limitations.
            return [torch.float16, torch.float32]
        return [torch.bfloat16, torch.float16, torch.float32]
