# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologies Co., Ltd.
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
    AclnnQuantBatchMatMul, ASDQuantBatchMatMul, Quant)
from vllm_mindspore.utils import is_310p


class A8W8LinearMethod(LinearMethodBase):

    def __init__(self, quant_config: QuantizationConfig) -> None:
        self.quant_config = quant_config
        self.is_modelslim = self.quant_config.is_modelslim
        self.is_310p = is_310p()

    def create_weights(self, layer: nn.Cell, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype, **extra_weight_attrs):
        output_size = sum(output_partition_sizes)
        self.output_size = output_size
        self.input_size_per_partition = input_size_per_partition
        self.params_dtype = params_dtype
        self.qbmm = AclnnQuantBatchMatMul(
            params_dtype
        ) if self.quant_config.is_modelslim else ASDQuantBatchMatMul(
            params_dtype)
        self.quant = Quant(mstype.int8)
        self.bias_add = ops.Add()

        weight_shape = (output_size, self.input_size_per_partition)
        weight = Parameter(initializer('ones', weight_shape, mstype.int8),
                           requires_grad=False)
        deq_scale = Parameter(initializer('ones', output_size, mstype.float32),
                              name="deq_scale",
                              requires_grad=False)
        quant_bias = Parameter(initializer('zeros', (output_size, ),
                                           mstype.int32),
                               name="quant_bias",
                               requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(deq_scale, {"output_dim": 0})
        set_weight_attrs(quant_bias, {"output_dim": 0})

        set_weight_attrs(weight, extra_weight_attrs)
        set_weight_attrs(deq_scale, extra_weight_attrs)
        set_weight_attrs(quant_bias, extra_weight_attrs)

        if not self.is_modelslim:
            input_scale = Parameter(initializer('ones',
                                                (input_size_per_partition, ),
                                                self.params_dtype),
                                    name="input_scale",
                                    requires_grad=False)
            input_offset = Parameter(initializer('zeros',
                                                 (input_size_per_partition, ),
                                                 mstype.int8),
                                     name="input_offset",
                                     requires_grad=False)
            set_weight_attrs(input_offset, {"input_dim": 0})
            set_weight_attrs(input_scale, {"input_dim": 0})
        else:
            input_scale = Parameter(initializer('ones', (1, ),
                                                self.params_dtype),
                                    name="input_scale",
                                    requires_grad=False)
            input_offset = Parameter(initializer('zeros', (1, ),
                                                 self.params_dtype),
                                     name="input_offset",
                                     requires_grad=False)
            beta = Parameter(initializer('zeros', (input_size_per_partition, ),
                                         self.params_dtype),
                             name="beta",
                             requires_grad=False)
            set_weight_attrs(beta, {"input_dim": 0})
            set_weight_attrs(beta, extra_weight_attrs)

        if isinstance(layer, RowParallelLinear):
            set_weight_attrs(input_scale, extra_weight_attrs)
            set_weight_attrs(input_offset, extra_weight_attrs)

        if layer is not None:
            layer.insert_param_to_cell("weight", weight)
            layer.insert_param_to_cell("deq_scale", deq_scale)
            layer.insert_param_to_cell("input_scale", input_scale)
            layer.insert_param_to_cell("input_offset", input_offset)
            layer.insert_param_to_cell("quant_bias", quant_bias)
            if self.is_modelslim:
                layer.insert_param_to_cell("beta", beta)

    def process_weights_after_loading(self, layer: nn.Cell) -> None:
        self.qbmm.process_weights_after_loading(layer)
        if self.is_310p:
            deq_scale = layer.deq_scale.asnumpy().view(np.int32).astype(
                np.int64)
            layer.deq_scale = Parameter(Tensor(deq_scale, dtype=mstype.int64),
                                        name=layer.deq_scale.name,
                                        requires_grad=False)

    def apply(self,
              layer: nn.Cell,
              x: Tensor,
              bias: Parameter = None) -> Tensor:
        weight = layer.weight
        deq_scale = layer.deq_scale
        input_scale = layer.input_scale
        input_offset = layer.input_offset
        quant_bias = layer.quant_bias
        if self.is_modelslim:
            beta = layer.beta
            x = ops.add(x, beta)
        qx = self.quant(x, input_scale, input_offset)
        output_shape = qx.shape[:-1] + (self.output_size, )
        qx = qx.reshape(-1, self.input_size_per_partition)
        out = self.qbmm(qx, weight, deq_scale, quant_bias)
        if bias is not None:
            out = self.bias_add(out, bias)
        out = out.reshape(output_shape)
        return out
