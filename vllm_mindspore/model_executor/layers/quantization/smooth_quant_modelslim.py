# SPDX-License-Identifier: Apache-2.0
#
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

from typing import Any, Optional

import mindspore
import numpy as np
import regex as re
from mindspore import Parameter, Tensor, mint
from mindspore.common.initializer import initializer
from mindspore.ops.auto_generate import (DynamicQuantExt, GroupedMatmul,
                                         GroupedMatmulV4, QuantBatchMatmul)
from mindspore.ops.operations._infer_ops import QuantV2

from vllm_mindspore.model_executor.layers.linear import (
    LinearBase, LinearMethodBase, UnquantizedLinearMethod)
from vllm_mindspore.model_executor.utils import set_weight_attrs
from vllm_mindspore.utils import is_310p
from vllm_mindspore.v1.attention import Attention

from .attention import BaseKVCacheMethod, KVCacheInt8Method
from .base_config import QuantizationConfig, QuantizeMethodBase


class SmoothQuantModelSlimConfig(QuantizationConfig):
    """Config class for SmoothQuant."""

    def __init__(
        self,
        full_config: dict[str, Any],
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
                f"A8W8, but got {self.weight_bits} bits.")
        self.pack_factor = 8 // self.weight_bits

    def __repr__(self) -> str:
        return (f"A8W8Config(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"zero_point={self.zero_point}, "
                f"modules_to_not_convert={self.modules_to_not_convert})")

    def get_name(self) -> str:
        return "SmoothQuant"

    def get_supported_act_dtypes(self) -> list[mindspore.dtype]:
        return [mindspore.int8, mindspore.float16, mindspore.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        """Minimum GPU capability to support the quantization method.

        E.g., 70 for Volta, 75 for Turing, 80 for Ampere.
        This requirement is due to the custom CUDA kernels used by the
        quantization method.
        """
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quant_model_description.json"]

    @classmethod
    def from_config(cls, config: dict[str,
                                      Any]) -> "SmoothQuantModelSlimConfig":
        return cls(config)

    def get_quant_method(self, layer: mindspore.nn.Cell,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if _is_layer_skipped_a8w8(prefix, self.modules_to_not_convert):
            return UnquantizedLinearMethod()
        quant_key = _build_layer_quant_key(prefix)
        quant_config = self.full_config.get(quant_key)
        if isinstance(layer, Attention):
            if layer.cache_config.cache_dtype == "int8" and quant_config \
                and quant_config.lower() == 'c8':
                return KVCacheInt8Method(self)
            return BaseKVCacheMethod(self)

        if isinstance(layer, LinearBase):
            if quant_config and quant_config.lower() == 'w8a8':
                return A8W8LinearMethod(self)
            if quant_config and quant_config.lower() == 'w8a8_dyn':
                self.dynamic_quant = True
                return A8W8DYNLinearMethod(self)

        print(f"get_quant_method unmatched {layer.__class__.__name__}, "
              f"{quant_key,self.full_config.get(quant_key)}")
        return UnquantizedLinearMethod()


def _build_layer_quant_key(prefix: str) -> str:
    # Split the fused qkv projection into the standard q projection.
    prefix = prefix.replace("qkv_proj", "q_proj")
    # Collapse gate+up projection to the canonical gate projection.
    prefix = prefix.replace("gate_up_proj", "gate_proj")

    # If the path contains a bare "experts", inject the default expert index "0"
    if not re.search(r"experts\.\d+", prefix):
        prefix = re.sub(r"\bexperts\b", "experts.0", prefix)

    # Step 4: decide whether to add ".weight"
    proj_names = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    last_token = prefix.split(".")[-1]
    if last_token in proj_names and not prefix.endswith(".weight"):
        prefix += ".weight"

    if last_token == "attn":
        prefix = prefix[:-5] + ".k_proj.kv_cache_scale"

    return prefix


def _is_layer_skipped_a8w8(prefix: str, modules_to_not_convert: list[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)


class A8W8LinearMethod(LinearMethodBase):
    """Linear method for A8W8LinearMethod."""

    def __init__(self, quant_config: SmoothQuantModelSlimConfig):
        self.quant_config = quant_config
        self.quant = QuantV2()

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
        if self.quant_config.group_size and input_size_per_partition \
                % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        output_size_per_partition = sum(output_partition_sizes)
        self.output_size_per_partition = output_size_per_partition
        self.input_size_per_partition = input_size_per_partition
        self.expert_num_per_partition = expert_num_per_partition
        self.params_dtype = params_dtype
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        self.is_group_mm = is_group_mm
        self.is_310p = is_310p()
        self.gmm_transpose = self.is_310p
        if self.is_group_mm:
            self._gmm_create_weight(layer, extra_weight_attrs)
        else:
            self._qmm_create_weight(layer, extra_weight_attrs)

    def _qmm_create_weight(self, layer, extra_weight_attrs):
        self.matmul = QuantBatchMatmul(transpose_x1=False,
                                       transpose_x2=True,
                                       dtype=self.params_dtype)
        weight_shape = (self.output_size_per_partition //
                        self.quant_config.pack_factor,
                        self.input_size_per_partition)
        weight = Parameter(initializer('ones', weight_shape, mindspore.int8),
                           requires_grad=False)
        weight_scale_shape = (self.output_size_per_partition //
                              self.quant_config.pack_factor, 1)
        scale_dtype = mindspore.bfloat16 if self.params_dtype == \
                        mindspore.bfloat16 else mindspore.float32
        weight_scale = Parameter(initializer('ones', weight_scale_shape,
                                             scale_dtype),
                                 name="weight_scale",
                                 requires_grad=False)
        deq_scale_shape = (self.output_size_per_partition //
                           self.quant_config.pack_factor)
        scale_dtype = mindspore.int64
        deq_scale = Parameter(initializer('ones', deq_scale_shape,
                                          scale_dtype),
                              name="deq_scale",
                              requires_grad=False)
        input_scale_shape = (1, )
        input_scale = Parameter(initializer('ones', input_scale_shape,
                                            self.params_dtype),
                                name="input_scale",
                                requires_grad=False)
        input_offset = Parameter(initializer('zeros', input_scale_shape,
                                             self.params_dtype),
                                 name="input_offset",
                                 requires_grad=False)
        if self.is_310p:
            quant_bias_ = Parameter(initializer(
                'zeros', (self.output_size_per_partition //
                          self.quant_config.pack_factor, ), mindspore.int32),
                                    name="quant_bias_",
                                    requires_grad=False)
        else:
            quant_bias_ = None

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight_scale, {"output_dim": 0})
        set_weight_attrs(deq_scale, {"output_dim": 0})

        set_weight_attrs(weight, extra_weight_attrs)
        set_weight_attrs(weight_scale, extra_weight_attrs)
        set_weight_attrs(deq_scale, extra_weight_attrs)
        set_weight_attrs(input_scale, extra_weight_attrs)
        set_weight_attrs(input_offset, extra_weight_attrs)
        if quant_bias_ is not None:
            set_weight_attrs(quant_bias_, extra_weight_attrs)
            layer.insert_param_to_cell("quant_bias_", quant_bias_)
        else:
            layer.quant_bias_ = None

        layer.insert_param_to_cell("weight", weight)
        layer.insert_param_to_cell("weight_scale", weight_scale)
        layer.insert_param_to_cell("deq_scale", deq_scale)
        layer.insert_param_to_cell("input_scale", input_scale)
        layer.insert_param_to_cell("input_offset", input_offset)

    def _gmm_create_weight(self, layer, extra_weight_attrs):
        if self.is_310p:
            self.matmul = GroupedMatmul(split_item=3,
                                        group_type=0,
                                        transpose_a=False,
                                        transpose_b=True)
        else:
            self.matmul = GroupedMatmulV4()
        shape = (self.expert_num_per_partition, self.input_size_per_partition,
                 self.output_size_per_partition //
                 self.quant_config.pack_factor)
        if self.gmm_transpose:
            shape = (shape[0], shape[2], shape[1])
        weight = Parameter(initializer('ones', shape, mindspore.int8),
                           requires_grad=False)
        weight_scale_shape = (self.expert_num_per_partition,
                              self.output_size_per_partition //
                              self.quant_config.pack_factor)
        scale_dtype = mindspore.bfloat16 if self.params_dtype == \
                        mindspore.bfloat16 else mindspore.float32
        weight_scale = Parameter(initializer('ones', weight_scale_shape,
                                             scale_dtype),
                                 name="weight_scale",
                                 requires_grad=False)
        scale_dtype = mindspore.int64
        deq_scale = Parameter(initializer('ones', weight_scale_shape,
                                          scale_dtype),
                              name="deq_scale",
                              requires_grad=False)
        input_scale_shape = (self.expert_num_per_partition, )
        input_scale = Parameter(initializer('ones', input_scale_shape,
                                            self.params_dtype),
                                name="input_scale",
                                requires_grad=False)
        input_offset = Parameter(initializer('zeros', input_scale_shape,
                                             self.params_dtype),
                                 name="input_offset",
                                 requires_grad=False)
        quant_bias_ = None
        set_weight_attrs(weight, {
            "ep_dim": 0,
            "input_dim": 1,
            "output_dim": 2
        })
        set_weight_attrs(weight_scale, {"ep_dim": 0, "output_dim": 1})
        set_weight_attrs(deq_scale, {"ep_dim": 0, "output_dim": 1})
        set_weight_attrs(input_scale, {"ep_dim": 0})
        set_weight_attrs(input_offset, {"ep_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        set_weight_attrs(weight_scale, extra_weight_attrs)
        set_weight_attrs(deq_scale, extra_weight_attrs)
        set_weight_attrs(input_scale, extra_weight_attrs)
        set_weight_attrs(input_offset, extra_weight_attrs)
        if quant_bias_ is not None:
            set_weight_attrs(quant_bias_, extra_weight_attrs)
            layer.insert_param_to_cell("quant_bias_", quant_bias_)
        else:
            layer.quant_bias_ = None

        layer.insert_param_to_cell("weight", weight)
        layer.insert_param_to_cell("weight_scale", weight_scale)
        layer.insert_param_to_cell("deq_scale", deq_scale)
        layer.insert_param_to_cell("input_scale", input_scale)
        layer.insert_param_to_cell("input_offset", input_offset)

    def process_weights_after_loading(self, layer: mindspore.nn.Cell) -> None:
        input_offset = np.array([0])
        params_dtype = layer.params_dtype
        layer.input_offset = Parameter(Tensor(input_offset,
                                              dtype=mindspore.int8),
                                       name=layer.input_offset.name,
                                       requires_grad=False)
        if self.is_group_mm:
            input_scale = layer.input_scale.asnumpy()
            weight_scale = layer.weight_scale.asnumpy()
            input_scale = np.array([np.max(input_scale)])
            weight_scale = weight_scale * input_scale[0]
            layer.input_scale = Parameter(Tensor(
                input_scale, dtype=layer.input_scale.dtype),
                                          name=layer.input_scale.name,
                                          requires_grad=False)
            if self.is_310p:
                layer.weight_scale = Parameter(Tensor(weight_scale.view(
                    np.int32).astype(np.int64),
                                                      dtype=mindspore.int64),
                                               name=layer.weight_scale.name,
                                               requires_grad=False)
            else:
                layer.weight_scale = Parameter(Tensor(
                    weight_scale, dtype=layer.weight_scale.dtype),
                                               name=layer.weight_scale.name,
                                               requires_grad=False)
        if not self.is_310p and params_dtype is mindspore.bfloat16:
            deq_scale = layer.deq_scale.asnumpy().astype(np.int32).view(
                np.float32)
            layer.deq_scale = Parameter(Tensor(deq_scale,
                                               dtype=mindspore.float32),
                                        name=layer.deq_scale.name,
                                        requires_grad=False)

    def apply(self,
              layer: mindspore.nn.Cell,
              x: mindspore.Tensor,
              bias: mindspore.Parameter = None,
              group_list=None,
              cumsum_flag=False) -> mindspore.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        deq_scale = layer.deq_scale
        input_scale = layer.input_scale
        input_offset = layer.input_offset
        qx = self.quant(x, input_scale, input_offset, False, "ROUND",
                        mindspore.dtype.int8)
        output_shape = qx.shape[:-1] + (self.output_size_per_partition, )
        qx = qx.reshape(-1, self.input_size_per_partition)
        if self.is_group_mm:
            if self.is_310p:
                qx = self.matmul([qx], [weight], None, [weight_scale], None,
                                 None, None, group_list.to(mindspore.int32))[0]
            else:
                qx = self.matmul([qx], [weight],
                                 None, [weight_scale],
                                 None,
                                 None,
                                 None,
                                 None,
                                 group_list,
                                 split_item=3,
                                 group_type=0,
                                 group_list_type=0 if cumsum_flag else 1)[0]
        else:
            qx = self.matmul(qx, weight, deq_scale, None, layer.quant_bias_,
                             None)
        if bias is not None:
            qx = mint.add(qx, bias)
        qx = qx.reshape(output_shape)
        return qx


class A8W8DYNLinearMethod(LinearMethodBase):
    """Linear method for A8W8DYNLinearMethod."""

    def __init__(self, quant_config: SmoothQuantModelSlimConfig):
        self.quant_config = quant_config
        self.quant = DynamicQuantExt()

    def create_weights(self,
                       layer: mindspore.nn.Cell,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int],
                       input_size: int,
                       output_size: int,
                       params_dtype,
                       is_group_mm=False,
                       expert_num_per_partition=1,
                       is_2d_smooth_scale=False,
                       **extra_weight_attrs):
        if self.quant_config.group_size and input_size_per_partition % \
            self.quant_config.group_size != 0:
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
        self.is_group_mm = is_group_mm
        self.gmm_transpose = False
        self.is_2d_smooth_scale = is_2d_smooth_scale

        if self.is_group_mm:
            self.matmul = GroupedMatmulV4()
            shape = (expert_num_per_partition, input_size_per_partition,
                     output_size_per_partition //
                     self.quant_config.pack_factor)
            weight = Parameter(initializer('ones', shape, mindspore.int8),
                               requires_grad=False)
            gmm_weight_scale_shape = (expert_num_per_partition,
                                      output_size_per_partition //
                                      self.quant_config.pack_factor)
            weight_scale_dtype = mindspore.bfloat16 if params_dtype == \
                                mindspore.bfloat16 else mindspore.float32
            weight_scale = Parameter(initializer('ones',
                                                 gmm_weight_scale_shape,
                                                 weight_scale_dtype),
                                     requires_grad=False)

            smooth_scale_shape = (input_size_per_partition, )
            smooth_scale = Parameter(initializer('ones', smooth_scale_shape,
                                                 params_dtype),
                                     requires_grad=False)
            set_weight_attrs(smooth_scale, {"input_dim": 0})

            set_weight_attrs(weight, {
                "ep_dim": 0,
                "input_dim": 1,
                "output_dim": 2
            })
            set_weight_attrs(weight_scale, {"ep_dim": 0, "output_dim": 1})
        else:
            self.matmul = QuantBatchMatmul(transpose_x1=False,
                                           transpose_x2=True,
                                           dtype=params_dtype)
            weight_shape = (output_size_per_partition //
                            self.quant_config.pack_factor,
                            input_size_per_partition)
            weight = Parameter(initializer('ones', weight_shape,
                                           mindspore.int8),
                               requires_grad=False)
            weight_scale_shape = (output_size_per_partition //
                                  self.quant_config.pack_factor)
            weight_scale_dtype = mindspore.bfloat16 if params_dtype == \
                                mindspore.bfloat16 else mindspore.float32
            weight_scale = Parameter(initializer('ones', weight_scale_shape,
                                                 weight_scale_dtype),
                                     requires_grad=False)
            smooth_scale_shape = (input_size_per_partition, )
            smooth_scale = Parameter(initializer('ones', smooth_scale_shape,
                                                 params_dtype),
                                     requires_grad=False)

            set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
            set_weight_attrs(weight_scale, {"output_dim": 0})
            set_weight_attrs(smooth_scale, {"input_dim": 0})

        set_weight_attrs(weight, extra_weight_attrs)
        set_weight_attrs(weight_scale, extra_weight_attrs)
        set_weight_attrs(smooth_scale, extra_weight_attrs)

        layer.insert_param_to_cell("weight", weight)
        layer.insert_param_to_cell("weight_scale", weight_scale)
        layer.insert_param_to_cell("smooth_scale", smooth_scale)

    def process_weights_after_loading(self, layer: mindspore.nn.Cell) -> None:
        if self.is_2d_smooth_scale:
            smooth_scale = layer.smooth_scale.asnumpy().reshape(1, -1)
            layer.smooth_scale = Parameter(Tensor(smooth_scale,
                                                  dtype=mindspore.float32),
                                           name=layer.smooth_scale.name,
                                           requires_grad=False)

    def apply(self,
              layer: mindspore.nn.Cell,
              x: mindspore.Tensor,
              bias: mindspore.Parameter = None,
              group_list=None,
              cumsum_flag=False) -> mindspore.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        smooth_scale = layer.smooth_scale

        qx, qx_scale = self.quant(x, smooth_scale)
        qx_scale = qx_scale.reshape(-1)
        output_shape = qx.shape[:-1] + (self.output_size_per_partition, )
        qx = qx.reshape(-1, self.input_size_per_partition)
        if self.is_group_mm:
            qx = self.matmul([qx], [weight],
                             None, [weight_scale],
                             None,
                             None,
                             None, [qx_scale],
                             group_list,
                             split_item=3,
                             group_type=0,
                             group_list_type=0 if cumsum_flag else 1)[0]
        else:
            qx = self.matmul(qx, weight, weight_scale, None, None, qx_scale)
        if bias is not None:
            qx = mint.add(qx, bias)
        qx = qx.reshape(output_shape)
        return qx
