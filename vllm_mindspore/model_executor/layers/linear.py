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

from typing import List, Optional
from abc import abstractmethod

import numpy as np
import mindspore as ms
from mindspore import mint, ops, Tensor
from mindspore import Parameter

from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm_mindspore.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm_mindspore.model_executor.utils import set_weight_attrs
from vllm_mindspore.distributed.communication_op import ReduceFromModelParallelRegion


WEIGHT_LOADER_V2_SUPPORTED = [
    "CompressedTensorsLinearMethod",
    "AWQMarlinLinearMethod",
    "AWQLinearMethod",
    "GPTQMarlinLinearMethod",
    "Fp8LinearMethod",
    "MarlinLinearMethod",
    "QQQLinearMethod",
    "GPTQMarlin24LinearMethod",
    "TPUInt8LinearMethod",
    "GPTQLinearMethod",
    "FBGEMMFp8LinearMethod",
    "ModelOptFp8LinearMethod",
    "IPEXAWQLinearMethod",
    "IPEXGPTQLinearMethod",
    "HQQMarlinMethod",
]


class LinearMethodBase(QuantizeMethodBase):
    """Base class for different (maybe quantized) linear methods."""

    @abstractmethod
    def create_weights(
        self,
        layer: ms.nn.Cell,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype,
        **extra_weight_attrs
    ):
        """Create weights for a linear layer.
           The weights will be set as attributes of the layer.

        Args:
            layer: The layer that is using the LinearMethodBase factory.
            input_size_per_partition: Size of the weight input dim on rank X.
            output_partition_sizes: Sizes of the output dim of each logical
                weight on rank X. E.g., output_partition_sizes for QKVLinear
                is a list contains the width of Wq, Wk, Wv on rank X.
            input_size: Size of the input dim of the weight across all ranks.
            output_size: Size of the output dim of the weight across all ranks.
            params_dtype: Datatype of the parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self, layer: ms.nn.Cell, x: ms.Tensor, bias: Optional[ms.Tensor] = None
    ) -> ms.Tensor:
        """Apply the weights in layer to the input tensor.
        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(
        self,
        layer: ms.nn.Cell,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype,
        **extra_weight_attrs
    ):
        weight = Parameter(
            mint.zeros(
                (int(sum(output_partition_sizes)), int(input_size_per_partition)),
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        self.input_size_per_partition = int(input_size_per_partition)
        self.output_size_per_partition = int(sum(output_partition_sizes))
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        # layer.register_parameter("weight", weight)
        layer.insert_param_to_cell("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        self.matmul = ops.MatMul(transpose_b=True)
        self.bias_add = ops.Add()

    def apply(self,
              layer: ms.nn.Cell,
              x: Tensor,
              bias: Parameter = None):
        output_shape = x.shape[:-1] + (self.output_size_per_partition,)
        x = x.reshape(-1, self.input_size_per_partition)
        x = self.matmul(x, layer.weight)
        if bias is not None:
            x = self.bias_add(x, bias)
        x = x.reshape(output_shape)
        return x


class LinearBase(ms.nn.Cell):
    """Base linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype=None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            # params_dtype = torch.get_default_dtype()
            params_dtype = ms.float16
        self.params_dtype = params_dtype
        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = UnquantizedLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        raise NotImplementedError

    def weight_loader(self):
        return None


class ColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype=None,
        quant_config: Optional[QuantizationConfig] = None,
        output_sizes: Optional[List[int]] = None,
        prefix: str = "",
    ):
        super().__init__(
            input_size, output_size, skip_bias_add, params_dtype, quant_config, prefix
        )

        self.gather_output = gather_output

        # Divide the weight matrix along the last dimension.
        tp_size = get_tensor_model_parallel_world_size()
        assert self.quant_method is not None
        self.output_size_per_partition = divide(self.output_size, tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, tp_size) for output_size in self.output_sizes
            ]

        if output_sizes is None:
            output_sizes = [output_size]
        # vllm 中变量名称与megatron相似度高， 然后mindspeed 与 megatron的变量命名相似度高， 所以以vllm为基准， 可以最大可能保证变量命名一致性， 方便load。
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )
        if bias:
            self.bias = Parameter(
                mint.zeros(self.output_size_per_partition, dtype=params_dtype)
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.bias = None
            # self.register_parameter("bias", None)

    def construct(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self, input_, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def weight_loader(self, param, loaded_weight):
        tp_rank = get_tensor_model_parallel_rank()
        output_dim = getattr(param, "output_dim", None)

        # # Special case for GGUF
        # is_gguf_weight = getattr(param, "is_gguf_weight", False)
        # is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        # if is_gguf_weight_type:
        #     param.weight_type = loaded_weight.item()

        # # Materialize GGUF UninitializedParameter
        # if is_gguf_weight and isinstance(param, UninitializedParameter):
        #     param.materialize(loaded_weight.shape, dtype=loaded_weight.dtype)

        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

        # bitsandbytes loads the weights of the specific portion
        # no need to narrow here
        if output_dim is not None and not use_bitsandbytes_4bit:
            shard_size = param.shape[output_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size).contiguous()

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param.shape == loaded_weight.shape
        # param_data.copy_(loaded_weight)
        # param.set_data(loaded_weight)
        # param[:, start_idx:start_idx + shard_size] = loaded_weight
        param.set_data(loaded_weight)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.

    Args:
        input_size: input dimension of the linear layer.
        output_sizes: list of output dimensions of the linear layer.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make the output
                       available to all GPUs, otherwise, every GPU will have
                       its own output.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype=None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        self.output_sizes = output_sizes
        tp_size = get_tensor_model_parallel_world_size()
        assert all(output_size % tp_size == 0 for output_size in output_sizes)
        super().__init__(
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
        )

    def weight_loader(
        self, param, loaded_weight, loaded_shard_id: Optional[int] = None
    ):
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        assert loaded_shard_id < len(self.output_sizes)
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        if output_dim is not None:
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // tp_size
            shard_size = self.output_sizes[loaded_shard_id] // tp_size
            # # Special case for quantization.
            # # If quantized, we need to adjust the offset and size to account
            # # for the packing.
            # packed_dim = getattr(param, "packed_dim", None)
            # if packed_dim == output_dim:
            #     shard_size = shard_size // param.pack_factor
            #     shard_offset = shard_offset // param.pack_factor
            #     # Special case for Marlin.
            #     shard_size, shard_offset = adjust_marlin_shard(
            #         param, shard_size, shard_offset)

            # use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit",
            #                                 False)
            # if use_bitsandbytes_4bit:
            #     shard_size = loaded_weight.shape[output_dim]
            #     shard_offset = loaded_weight.shape[output_dim] * \
            #         loaded_shard_id

            param_data = param.data
            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            start_idx = tp_rank * shard_size
            # bitsandbytes loads the weights of the specific portion
            # no need to narrow here
            if not use_bitsandbytes_4bit:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size).contiguous()
            assert param_data.shape == loaded_weight.shape
            # param_data.copy_(loaded_weight)
            # param_data.set_data(loaded_weight)
            param[shard_offset: shard_offset + shard_size, :] = loaded_weight


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype=None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = (
            (self.num_heads + 2 * self.num_kv_heads) * tp_size * self.head_size
        )
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.head_size * tp_size,  # v_proj
        ]

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
        )

    def weight_loader(self, param, loaded_weight, loaded_shard_id):
        output_dim = getattr(param, "output_dim", None)
        tp_rank = get_tensor_model_parallel_rank()
        assert loaded_shard_id in ["q", "k", "v"]
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        # If output dim is defined, use the default loading process.
        # if output_dim is not None:
        param_data = param.data
        if True:
            if loaded_shard_id == "q":
                shard_offset = 0
                shard_size = self.num_heads * self.head_size
            elif loaded_shard_id == "k":
                shard_offset = self.num_heads * self.head_size
                shard_size = self.num_kv_heads * self.head_size
            elif loaded_shard_id == "v":
                shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
                shard_size = self.num_kv_heads * self.head_size

            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            if loaded_shard_id == "q":
                shard_id = tp_rank
            else:
                shard_id = tp_rank // self.num_kv_head_replicas
            start_idx = shard_id * shard_size

            if not use_bitsandbytes_4bit:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size).contiguous()
            assert param_data.shape == loaded_weight.shape
            if param.name.endswith("weight"):
                self.weight[shard_offset: shard_offset + shard_size, :] = loaded_weight
            if param.name.endswith("bias"):
                self.bias[shard_offset: shard_offset + shard_size] = loaded_weight
        # tp_rank = get_tensor_model_parallel_rank()
        # if shard_id is "q":
        #     start_index = self.num_heads * tp_rank * self.head_size
        #     param_data = param_data.narrow(output_dim, start_index, self.num_heads * self.head_size)
        # else:
        #     start_index = self.num_kv_heads * tp_rank * self.head_size
        #     param_data = param_data.narrow(output_dim, start_index, self.kv_num_heads * self.head_size)

        # if shard_id is "q":
        #     self.weight[:, :self.num_heads * self.head_size] = param_data
        # elif shard_id is "k":
        #     self.weight[:, self.num_heads * self.head_size : self.num_kv_heads * self.head_size] = param_data
        # elif shard_id is "v":
        #     self.weight[:, (self.num_heads + self.num_kv_heads) * self.head_size :self.num_kv_heads * 2 * self.head_size] = param_data


class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype=None,
        reduce_results: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(
            input_size, output_size, skip_bias_add, params_dtype, quant_config, prefix
        )

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        # Divide the weight matrix along the last dimension.
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, self.tp_size)
        assert self.quant_method is not None

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=[self.output_size],
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError(
                "When not reduce the results, adding bias to the "
                "results can lead to incorrect results"
            )

        if bias:
            self.bias = Parameter(mint.zeros(self.output_size, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            # self.register_parameter("bias", None)
            self.bias = None

        self.tensor_model_parallel_all_reduce = ReduceFromModelParallelRegion()

    def construct(self, input_):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = splitted_input[tp_rank].contiguous()

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self, input_parallel, bias=bias_)
        if self.reduce_results and self.tp_size > 1:
            output = self.tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias

    def weight_loader(self, param, loaded_weight):
        tp_rank = get_tensor_model_parallel_rank()
        input_dim = getattr(param, "input_dim", None)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

        # bitsandbytes loads the weights of the specific portion
        # no need to narrow here
        if input_dim is not None and not use_bitsandbytes_4bit:
            shard_size = param.shape[input_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size).contiguous()

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param.shape == loaded_weight.shape
        # param_data.copy_(loaded_weight)
        # self.weight[:, start_idx : start_idx + shard_size] = loaded_weight
        param.set_data(loaded_weight.contiguous())
