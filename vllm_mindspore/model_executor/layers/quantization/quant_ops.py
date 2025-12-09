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

from mindspore import Parameter, Tensor, nn, ops
from mindspore.common import dtype as mstype

ND_TO_FRACTAL_NZ = 1


class Quant(nn.Cell):
    """
    This module performs per-tensor quantization on the input activation
    using provided scale and offset parameters.

    Args:
        quant_dtype: Target dtype after quantization (e.g., mstype.int8).

    Inputs:
        x (Tensor): Float activation tensor to be quantized.
        input_scale (Parameter): Scale factor for quantization.
        input_offset (Parameter): Zero-point (offset) for quantization.

    Returns:
        Tensor: Quantized tensor of dtype `quant_dtype`.
    """

    def __init__(self, quant_dtype):
        super().__init__()
        self.quant = ops.auto_generate.QuantV2()
        self.quant_dtype = quant_dtype

    def construct(self, x: Tensor, input_scale: Parameter,
                  input_offset: Parameter) -> Tensor:
        qx = self.quant(x, input_scale, input_offset, False, "ROUND",
                        self.quant_dtype)
        return qx


class ASDQuantBatchMatMul(nn.Cell):
    """
    Quantized batch matrix multiplication wrapper for ASD.
    
    Args:
        params_dtype: Output dtype (usually bfloat16).

    Inputs:
        qx (Tensor): Quantized input activation.
        weight (Parameter): Quantized weight parameter.
        deq_scale (Parameter): Dequantization scale for output.
        quant_bias (Parameter): Quantized bias (may be None).

    Returns:
        Tensor: Output tensor after quantized batch matmul.
    """

    def __init__(self, params_dtype):
        super().__init__()
        import ms_custom_ops
        self.trans_data = ms_custom_ops.trans_data
        self.quant_batch_matmul = ms_custom_ops.quant_batch_matmul
        self.params_dtype = params_dtype

    def process_weights_after_loading(self, layer: nn.Cell) -> None:
        """ASDQuantBatchMatMul requires weight to be in FRACTAL_NZ format."""
        layer.weight = self.trans_data(layer.weight,
                                       transdata_type=ND_TO_FRACTAL_NZ)

    def construct(self, qx: Tensor, weight: Parameter, deq_scale: Parameter,
                  quant_bias: Parameter) -> Tensor:
        """Apply quantized matrix multiplication."""
        return self.quant_batch_matmul(qx,
                                       weight,
                                       deq_scale,
                                       None,
                                       quant_bias,
                                       None,
                                       transpose_x1=False,
                                       transpose_x2=True,
                                       x2_format="FRACTAL_NZ",
                                       output_dtype=self.params_dtype)


class AclnnQuantBatchMatMul(nn.Cell):
    """
    Quantized batch matrix multiplication wrapper for ACLNN.
    
    Args:
        params_dtype: Output dtype (usually bfloat16).

    Inputs:
        qx (Tensor): Quantized input activation.
        weight (Parameter): Quantized weight parameter.
        deq_scale (Parameter): Dequantization scale for output.
        quant_bias (Parameter): Quantized bias (may be None).

    Returns:
        Tensor: Output tensor after quantized batch matmul.
    """

    def __init__(self, params_dtype):
        self.params_dtype = params_dtype
        self.quant_batch_matmul = ops.auto_generate.QuantBatchMatmul(
            transpose_x2=True, dtype=params_dtype)

    def process_weights_after_loading(self, layer: nn.Cell) -> None:
        """Process input_scale by taking reciprocal."""
        np_param_recip = 1 / layer.input_scale.numpy()
        layer.input_scale = Parameter(Tensor(np_param_recip,
                                             dtype=mstype.bfloat16),
                                      name=layer.input_scale.name,
                                      requires_grad=False)

    def construct(self, qx: Tensor, weight: Parameter, deq_scale: Parameter,
                  quant_bias: Parameter) -> Tensor:
        """Apply quantized matrix multiplication."""
        return self.quant_batch_matmul(qx, weight, deq_scale, None, quant_bias)
