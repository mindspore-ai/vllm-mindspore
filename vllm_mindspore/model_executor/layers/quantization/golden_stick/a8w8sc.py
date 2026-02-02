# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2025-2026 Huawei Technologies Co., Ltd.
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

import numpy as np
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs

from vllm_mindspore.model_executor.layers.linear import RowParallelLinear
from vllm_mindspore.model_executor.layers.quantization.quant_ops import (
    Quant, QuantLinearSparseOp)
from vllm_mindspore.utils import is_310p


class A8W8SCLinearMethod(LinearMethodBase):

    def __init__(self,
                 quant_config: QuantizationConfig,
                 compress_weight_size=None,
                 compress_index_size=None):
        self.quant_config = quant_config
        self.compress_weight_size = compress_weight_size
        self.compress_index_size = compress_index_size
        self.is_310p = is_310p()
        # Get weight dtype from config, default to int8 for W8A8SC
        self.weight_dtype = getattr(quant_config, 'weight_dtype', mstype.int8)
        self.index_dtype = getattr(quant_config, 'index_dtype', mstype.int8)
        self.quant = Quant(self.weight_dtype)

    def create_weights(self, layer: nn.Cell, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype, **extra_weight_attrs):
        output_size_per_partition = sum(output_partition_sizes)
        self.output_size_per_partition = output_size_per_partition
        self.input_size_per_partition = input_size_per_partition
        self.params_dtype = params_dtype
        self.linear_sparse = QuantLinearSparseOp(params_dtype)

        # Create compressed weight - no sharding for sparse quantized weights
        # Use dtype from config, default to int8 for W8A8SC
        weight = Parameter(initializer('normal',
                                       tuple(self.compress_weight_size),
                                       self.weight_dtype),
                           requires_grad=False)
        # Create sparse index - no sharding
        # Use dtype from config, default to int8 for W8A8SC
        index = Parameter(initializer('normal',
                                      tuple(self.compress_index_size),
                                      self.index_dtype),
                          requires_grad=False)
        # Dequantization scale (int64)
        deq_scale = Parameter(initializer('ones',
                                          (output_size_per_partition, ),
                                          mstype.int64),
                              name="deq_scale",
                              requires_grad=False)
        # Quantization bias (int32)
        quant_bias = Parameter(initializer('zeros',
                                           (output_size_per_partition, ),
                                           mstype.int32),
                               name="quant_bias",
                               requires_grad=False)
        # Input quantization parameters
        input_scale = Parameter(Tensor(np.ones(input_size_per_partition),
                                       mstype.float16),
                                name="input_scale",
                                requires_grad=False)
        input_offset = Parameter(Tensor(np.zeros(input_size_per_partition),
                                        mstype.int8),
                                 name="input_offset",
                                 requires_grad=False)

        # Do NOT set output_dim/input_dim or weight_loader
        # for sparse quantized weights
        # Sparse quantized weights are already partitioned by rank folders
        # and should be loaded directly without any sharding operations
        # Apply extra_weight_attrs but exclude output_dim/input_dim
        # to prevent sharding
        filtered_attrs = {
            k: v
            for k, v in extra_weight_attrs.items()
            if k not in ("output_dim", "input_dim", "weight_loader")
        }
        if filtered_attrs:
            set_weight_attrs(weight, filtered_attrs)
            set_weight_attrs(deq_scale, filtered_attrs)
            set_weight_attrs(quant_bias, filtered_attrs)
            set_weight_attrs(input_scale, filtered_attrs)
            set_weight_attrs(input_offset, filtered_attrs)

        if isinstance(layer, RowParallelLinear) and filtered_attrs:
            set_weight_attrs(input_scale, filtered_attrs)
            set_weight_attrs(input_offset, filtered_attrs)

        if layer is not None:
            layer.insert_param_to_cell("weight", weight)
            layer.insert_param_to_cell("index", index)
            layer.insert_param_to_cell("deq_scale", deq_scale)
            layer.insert_param_to_cell("quant_bias", quant_bias)
            layer.insert_param_to_cell("input_scale", input_scale)
            layer.insert_param_to_cell("input_offset", input_offset)

    def process_weights_after_loading(self, layer: nn.Cell) -> None:
        input_scale = layer.input_scale.asnumpy()
        input_offset = layer.input_offset.asnumpy()

        # Expand scalar scale/offset to full dimensions
        # Process input_scale and input_offset independently
        if input_scale.shape == (1, ):
            input_scale = np.full(shape=self.input_size_per_partition,
                                  fill_value=input_scale[0])
            layer.input_scale = Parameter(Tensor(input_scale,
                                                 dtype=mstype.float16),
                                          name=layer.input_scale.name,
                                          requires_grad=False)

        if input_offset.shape == (1, ):
            input_offset = np.full(shape=self.input_size_per_partition,
                                   fill_value=input_offset[0])
            # Use index_dtype for input_offset (typically int8)
            layer.input_offset = Parameter(Tensor(input_offset,
                                                  dtype=self.index_dtype),
                                           name=layer.input_offset.name,
                                           requires_grad=False)

        # Note: Sparse quantized weights should NOT be converted to Nz format
        # The weight_loader already handles this
        # by loading weights without sharding

    def apply(self,
              layer: nn.Cell,
              x: Tensor,
              bias: Parameter = None) -> Tensor:
        weight = layer.weight
        index = layer.index
        deq_scale = layer.deq_scale
        quant_bias = layer.quant_bias
        input_scale = layer.input_scale
        input_offset = layer.input_offset

        qx = self.quant(x, input_scale, input_offset)
        output_shape = qx.shape[:-1] + (self.output_size_per_partition, )
        qx = qx.reshape(-1, self.input_size_per_partition)
        out = self.linear_sparse(qx, weight, deq_scale, index, quant_bias)
        if bias is not None:
            out = ops.add(out, bias)
        out = out.reshape(output_shape)
        return out
