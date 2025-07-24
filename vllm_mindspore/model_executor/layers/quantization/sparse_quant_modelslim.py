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

from typing import Any, Optional, Dict

import torch
import numpy as np
import mindspore

from mindspore.common.initializer import initializer
from mindspore import Parameter, ops, Tensor
from mindspore.ops.operations._infer_ops import QuantV2
from mindspore.communication import get_rank
from vllm_mindspore.model_executor.layers.linear import LinearMethodBase, UnquantizedLinearMethod, LinearBase

from .base_config import QuantizationConfig



class SparseQuantModelSlimConfig(QuantizationConfig):
    '''Config class for SparseQuant.'''

    def __init__(
        self,
        full_config: Dict[str, Any],
        weight_bits: Optional[int] = 8,
        group_size: Optional[int] = 1,
        zero_point: Optional[bool] = True,
        dynamic_quant: Optional[bool] = False,
        kv_cache_bits: Optional[int] = 16,
        modules_to_not_convert: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.full_config = full_config
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.dynamic_quant = dynamic_quant
        self.kv_cache_bits = kv_cache_bits
        self.modules_to_not_convert = modules_to_not_convert or []

        if self.weight_bits != 8:
            raise ValueError(
                "Currently, only 8-bit weight quantization is supported for "
                f"A8W8SC, but got {self.weight_bits} bits.")
        self.pack_factor = 8 // self.weight_bits

    def __repr__(self) -> str:
        return (f"SparseConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"zero_point={self.zero_point}, "
                f"modules_to_not_convert={self.modules_to_not_convert})")

    @staticmethod
    def get_config_filenames() -> list[str]:
        return [
            "quant_model_description.json"
        ]

    @classmethod
    def get_min_capability(cls) -> int:
        """Minimum GPU capability to support the quantization method.

        E.g., 70 for Volta, 75 for Turing, 80 for Ampere.
        This requirement is due to the custom CUDA kernels used by the
        quantization method.
        """
        return -1

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SparseQuantModelSlimConfig":
        return cls(config)

    def get_name(self) -> str:
        return "SparseQuant"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.int8, torch.float16, torch.bfloat16]

    def get_quant_method(self, layer: mindspore.nn.Cell,
                         prefix: str) -> "QuantizeMethodBase":

        rank_id = get_rank()
        sparse_quant_description = self.full_config[f'rank_{rank_id}']
        if isinstance(layer, LinearBase) and sparse_quant_description[f"{prefix}.weight"].lower() == "w8a8s":
            compress_weight_size = sparse_quant_description[f"{prefix}.weight.shape"]
            compress_index_size = sparse_quant_description[f"{prefix}.index.shape"]

            return A8W8SCLinearMethod(self, compress_weight_size[0], compress_index_size[0])

        return UnquantizedLinearMethod()


class A8W8SCLinearMethod(LinearMethodBase):
    '''Linear method for A8W8SCLinearMethod.'''

    def __init__(self, quant_config: SparseQuantModelSlimConfig, compress_weight_size=None, compress_index_size=None):
        self.quant_config = quant_config
        self.compress_weight_size = compress_weight_size
        self.compress_index_size = compress_index_size

        self.quant = QuantV2()
        self.linear_sparse = ops.auto_generate.QuantLinearSparse()

    def create_weights(self,
            layer: mindspore.nn.Cell,
            input_size_per_partition: int,
            output_partition_sizes: list[int],
            input_size: int,
            output_size: int,
            params_dtype,
            is_group_mm=False,
            expert_num_per_partition=1,
            **extra_weight_attrs):
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        output_size_per_partition = sum(output_partition_sizes)
        self.output_size_per_partition = output_size_per_partition
        self.input_size_per_partition = input_size_per_partition
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        weight = Parameter(initializer('normal', (self.compress_weight_size), mindspore.int8), name="weight")
        index = Parameter(initializer('normal', (self.compress_index_size), mindspore.int8), name="index")
        deq_scale = Parameter(initializer('normal', (self.output_size_per_partition), mindspore.int64),
                              name="deq_scale")
        quant_bias = Parameter(initializer('zeros', (self.output_size_per_partition), mindspore.int32),
                               name="quant_bias")
        input_scale = Parameter(Tensor(np.ones(self.input_size_per_partition), mindspore.float16),
                                name="input_scale")
        input_offset = Parameter(Tensor(np.zeros(self.input_size_per_partition), mindspore.int8),
                                 name="input_offset")

        layer.insert_param_to_cell("weight", weight)
        layer.insert_param_to_cell("index", index)
        layer.insert_param_to_cell("deq_scale", deq_scale)
        layer.insert_param_to_cell("quant_bias", quant_bias)
        layer.insert_param_to_cell("input_scale", input_scale)
        layer.insert_param_to_cell("input_offset", input_offset)

    def apply(self,
              layer: mindspore.nn.Cell,
              x: mindspore.Tensor,
              bias: mindspore.Parameter = None, group_list=None, cumsum_flag=False) -> mindspore.Tensor:
        weight = layer.weight
        index = layer.index
        deq_scale = layer.deq_scale
        quant_bias = layer.quant_bias
        input_scale = layer.input_scale
        input_offset = layer.input_offset

        output_shape = x.shape[:-1] + (self.output_size_per_partition,)
        x = x.reshape(-1, self.input_size_per_partition)

        x = self.quant(x, input_scale, input_offset, False, "ROUND", mindspore.int8)
        x = self.linear_sparse(x, weight, deq_scale, index, quant_bias)

        x = x.reshape(output_shape)

        return x