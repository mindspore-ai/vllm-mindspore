# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/model_executor/model_loader/utils.py
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
""" utils for load model """
import numpy as np
import torch
from torch import nn
from vllm.attention import Attention
from vllm.config import ModelConfig
from vllm.model_executor.models import ModelRegistry

from vllm_mindspore.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase)
from vllm_mindspore.model_executor.models.registry import (
    MindSporeModelRegistry)


def get_ms_model_architecture(
        model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    architectures = getattr(model_config.hf_config, "architectures", [])

    vllm_supported_archs = ModelRegistry.get_supported_archs()
    is_vllm_supported = any(arch in vllm_supported_archs
                            for arch in architectures)
    if not is_vllm_supported:
        raise RuntimeError("vLLM-Mindspore does not support "
                           f"{str(architectures)} for now.")
    model_cls, arch = MindSporeModelRegistry.resolve_model_cls(architectures)
    if model_config.task == "embed":
        raise RecursionError("MindSpore unsupported embed model task now!")
    elif model_config.task == "classify":
        raise RecursionError("MindSpore unsupported classify model task now!")
    elif model_config.task == "reward":

        raise RecursionError("MindSpore unsupported reward model task now!")

    return model_cls, arch


def convert_uint64_to_fp32(arr: np.ndarray):
    arr_fp32 = arr.view(np.float32)
    output = arr_fp32[:, :, 0::2]
    return output


def np_int4data_pack_to_int8_3d(np_data):
    np_data = np_data.astype(np.int8)
    np_data &= 0x000F
    np_data[::, ::, 0::2] <<= 0
    np_data[::, ::, 1::2] <<= 4
    np_int4_data = np_data[::, ::, 0::2] | np_data[::, ::, 1::2]
    return np_int4_data


def unpack_int8_to_int4_3d(packed_data):
    low_nibbles = (packed_data & 0x0F).astype(np.uint8)
    high_nibbles = ((packed_data >> 4) & 0x0F).astype(np.uint8)

    unpacked = np.empty((*packed_data.shape[:2], packed_data.shape[2] * 2),
                        dtype=np.uint8)
    unpacked[..., 0::2] = low_nibbles
    unpacked[..., 1::2] = high_nibbles

    return unpacked


def process_weights_after_loading(model: nn.Module, model_config: ModelConfig,
                                  target_device: torch.device) -> None:
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            # # When quant methods need to process weights after loading
            # # (for repacking, quantizing, etc), they expect parameters
            # # to be on the global target device. This scope is for the
            # # case where cpu offloading is used, where we will move the
            # # parameters onto device for processing and back off after.
            # with device_loading_context(module, target_device):
            quant_method.process_weights_after_loading(module)

    # Currently only used by MLA.
    # NOTE: This intentionally happens after other modules so we can easily
    # decompress the weights for MLA.
    for _, module in model.named_modules():
        if isinstance(module, Attention) and \
            hasattr(module, "process_weights_after_loading"):
            # TODO(lucas): see if there is a way to unify the signatures
            # of process_weights_after_loading
            module.process_weights_after_loading(model_config.dtype)
