# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/model_executor/layers/linear.py
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
""" Linear methods for quantized linear layers. """

from abc import abstractmethod
from typing import Optional, Union

import mindspore as ms
import numpy as np
from mindspore import Parameter, Tensor, mint, nn, ops
from mindspore._c_expression.typing import Type as MSDtype
from vllm.config import get_current_vllm_config
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim)

from vllm_mindspore.distributed.communication_op import (
    AllGatherFromModelParallelRegion, ReduceFromModelParallelRegion)
from vllm_mindspore.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm_mindspore.model_executor.model_loader.weight_utils import (
    split_loaded_weight)
from vllm_mindspore.model_executor.utils import set_weight_attrs

WEIGHT_LOADER_V2_SUPPORTED = [
    "CompressedTensorsLinearMethod", "AWQMarlinLinearMethod",
    "AWQLinearMethod", "GPTQMarlinLinearMethod", "Fp8LinearMethod",
    "MarlinLinearMethod", "QQQLinearMethod", "GPTQMarlin24LinearMethod",
    "TPUInt8LinearMethod", "GPTQLinearMethod", "FBGEMMFp8LinearMethod",
    "ModelOptFp8LinearMethod", "IPEXAWQLinearMethod", "IPEXGPTQLinearMethod",
    "HQQMarlinMethod", "QuarkLinearMethod"
]


class LinearMethodBase(QuantizeMethodBase):
    """Base class for different (maybe quantized) linear methods."""

    @abstractmethod
    def create_weights(self, layer: nn.Cell, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: MSDtype,
                       **extra_weight_attrs):
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
    def apply(self,
              layer: nn.Cell,
              x: Tensor,
              bias: Optional[Tensor] = None) -> Tensor:
        """Apply the weights in layer to the input tensor.
        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def __init__(self):
        self.matmul = ops.MatMul(transpose_b=True)

    def create_weights(self, layer: nn.Cell, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype, **extra_weight_attrs):
        weight = Parameter(
            mint.zeros(
                (int(sum(output_partition_sizes)),
                 int(input_size_per_partition)),
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        self.input_size_per_partition = int(input_size_per_partition)
        self.output_size_per_partition = int(sum(output_partition_sizes))
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.insert_param_to_cell("weight", weight)

    def apply(self,
              layer: nn.Cell,
              x: Tensor,
              bias: Parameter = None) -> Tensor:
        output_shape = x.shape[:-1] + (self.output_size_per_partition, )
        x = x.view(-1, self.input_size_per_partition)
        x = self.matmul(x, layer.weight)
        if bias is not None:
            x = mint.add(x, bias)
        x = x.view(output_shape)
        return x


class LinearBase(nn.Cell):
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
        params_dtype: Optional[MSDtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: Optional[bool] = True,
    ):
        super().__init__()
        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = get_current_vllm_config().model_config.dtype
        self.params_dtype = params_dtype
        if quant_config is None:
            self.quant_method: Optional[
                QuantizeMethodBase] = UnquantizedLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self,
                                                              prefix=prefix)
        self.return_bias = return_bias

    def construct(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """Replicated linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype=None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 *,
                 return_bias: bool = True):
        super().__init__(input_size,
                         output_size,
                         skip_bias_add,
                         params_dtype,
                         quant_config,
                         prefix=prefix,
                         return_bias=return_bias)

        # All the linear layer supports quant method.
        assert self.quant_method is not None
        self.quant_method.create_weights(self,
                                         self.input_size, [self.output_size],
                                         self.input_size,
                                         self.output_size,
                                         self.params_dtype,
                                         weight_loader=self.weight_loader)

        if bias:
            self.bias = Parameter(
                mint.empty(self.output_size, dtype=self.params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.bias = None

    def weight_loader(self, param: Parameter, loaded_weight: Tensor):
        loaded_weight = loaded_weight[:]
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param.shape == loaded_weight.shape, (
            f"Tried to load weights of size {loaded_weight.size()}"
            f"to a parameter of size {param.size()}")

        param.set_data(ms.from_numpy(loaded_weight))

    def construct(
            self,
            x: Tensor) -> Union[Tensor, tuple[Tensor, Optional[Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None
        output = self.quant_method.apply(self, x, bias)
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        return s


class ColumnParallelLinear(LinearBase):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make Y available
                       to all NPUs, otherwise, every NPU will have its output
                       which is Y_i = XA_i
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        output_sizes: list of output sizes packed into one output, like for QKV
                       the list would be size 3.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[MSDtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        output_sizes: Optional[list[int]] = None,
        prefix: str = "",
        *,
        return_bias: Optional[bool] = True,
    ):
        super().__init__(input_size,
                         output_size,
                         skip_bias_add,
                         params_dtype,
                         quant_config,
                         prefix,
                         return_bias=return_bias)

        self.gather_output = gather_output

        # Divide the weight matrix along the last dimension.
        tp_size = get_tensor_model_parallel_world_size()
        assert self.quant_method is not None
        self.output_size_per_partition = divide(self.output_size, tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, tp_size)
                for output_size in self.output_sizes
            ]

        if output_sizes is None:
            output_sizes = [output_size]
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2 if self.quant_method.__class__.__name__
                in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader),
        )
        if bias:
            self.bias = Parameter(mint.zeros(self.output_size_per_partition,
                                             dtype=self.params_dtype),
                                  requires_grad=False)
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.bias = None

        self.tensor_model_parallel_all_gather = \
            AllGatherFromModelParallelRegion()

    def construct(self,
                  input_: Tensor) -> Union[Tensor, tuple[Tensor, Tensor]]:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self, input_, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = self.tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias

    def weight_loader(self, param, loaded_weight):
        tp_rank = get_tensor_model_parallel_rank()
        output_dim = getattr(param, "output_dim", None)
        shard_size = self.output_size_per_partition
        start_idx = tp_rank * shard_size
        loaded_weight = split_loaded_weight(loaded_weight, output_dim,
                                            start_idx, shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param.shape == loaded_weight.shape
        param.set_data(ms.from_numpy(loaded_weight))


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
                       available to all NPUs, otherwise, every NPU will have
                       its own output.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    def __init__(self,
                 input_size: int,
                 output_sizes: list[int],
                 bias: bool = True,
                 gather_output: bool = False,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[MSDtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 *,
                 return_bias: Optional[bool] = True):
        self.output_sizes = output_sizes
        tp_size = get_tensor_model_parallel_world_size()
        assert all(output_size % tp_size == 0 for output_size in output_sizes)
        super().__init__(input_size=input_size,
                         output_size=sum(output_sizes),
                         bias=bias,
                         gather_output=gather_output,
                         skip_bias_add=skip_bias_add,
                         params_dtype=params_dtype,
                         quant_config=quant_config,
                         prefix=prefix,
                         return_bias=return_bias)

    def weight_loader(self,
                      param,
                      loaded_weight,
                      loaded_shard_id: Optional[int] = None):
        output_dim = getattr(param, "output_dim", None)
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        shard_size = 0
        shard_offset = 0
        if loaded_shard_id is None:
            current_shard_offset = 0
            shard_offsets = []
            for i, output_size in enumerate(self.output_sizes):
                shard_offsets.append((i, current_shard_offset, output_size))
                current_shard_offset += output_size
            for shard_id, shard_offset, shard_size in shard_offsets:
                loaded_weight_shard = split_loaded_weight(
                    loaded_weight, output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
        else:
            assert loaded_shard_id < len(self.output_sizes)
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // tp_size
            shard_size = self.output_sizes[loaded_shard_id] // tp_size

            start_idx = tp_rank * shard_size
            loaded_weight = split_loaded_weight(loaded_weight, output_dim,
                                                start_idx, shard_size)

            assert loaded_weight.shape == (shard_size, param.shape[1])
            param[shard_offset:shard_offset +
                  shard_size, :] = ms.from_numpy(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.

    Args:
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_heads: total number of attention query heads.
        total_num_kv_heads: total number of attention key/value heads. If
                            None, assume total_num_kv_heads = total_num_heads.
        bias: If true, add bias.
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
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[MSDtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: Optional[bool] = True,
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
            self.num_kv_head_replicas = divide(tp_size,
                                               self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = ((self.num_heads + 2 * self.num_kv_heads) * tp_size *
                       self.head_size)
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.head_size * tp_size,  # v_proj
        ]

        super().__init__(input_size=input_size,
                         output_size=output_size,
                         bias=bias,
                         gather_output=False,
                         skip_bias_add=skip_bias_add,
                         params_dtype=params_dtype,
                         quant_config=quant_config,
                         prefix=prefix,
                         return_bias=return_bias)

    def weight_loader(self,
                      param,
                      loaded_weight,
                      loaded_shard_id: Optional[str] = None):
        output_dim = getattr(param, "output_dim", None)
        tp_rank = get_tensor_model_parallel_rank()

        # QKV loaded weight is already fused on disk (qkv safetensors).
        # According to the Safetensor of qkv, after partitioning,
        # load it into the corresponding qkv fusion weights
        if loaded_shard_id is None:
            shard_offsets = [
                # (shard_id, shard_offset, shard_size) # noqa: ERA001
                ("q", 0, self.num_heads * self.head_size),
                ("k", self.total_num_heads * self.head_size,
                 self.num_kv_heads * self.head_size),
                ("v", (self.total_num_heads + self.total_num_kv_heads) *
                 self.head_size, self.num_kv_heads * self.head_size),
            ]

            loaded_weight_list = []
            for _, shard_offset, shard_size in shard_offsets:
                start_idx = shard_offset + tp_rank * shard_size
                loaded_weight_shard = split_loaded_weight(
                    loaded_weight, output_dim, start_idx, shard_size)
                loaded_weight_list.append(loaded_weight_shard)

            loaded_weight = ms.from_numpy(np.concatenate(loaded_weight_list))

            assert loaded_weight.shape == param.shape
            param.set_data(loaded_weight)
            return

        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_offset = 0
            shard_size = self.num_heads * self.head_size
        elif loaded_shard_id == "k":
            shard_offset = self.num_heads * self.head_size
            shard_size = self.num_kv_heads * self.head_size
        elif loaded_shard_id == "v":
            shard_offset = (self.num_heads +
                            self.num_kv_heads) * self.head_size
            shard_size = self.num_kv_heads * self.head_size

        if loaded_shard_id == "q":
            shard_id = tp_rank
        else:
            shard_id = tp_rank // self.num_kv_head_replicas
        start_idx = shard_id * shard_size
        loaded_weight = split_loaded_weight(loaded_weight, output_dim,
                                            start_idx, shard_size)
        loaded_weight = ms.from_numpy(loaded_weight)

        if param.name.endswith("weight"):
            assert loaded_weight.shape == (shard_size, param.shape[1])
        if param.name.endswith("bias"):
            assert loaded_weight.shape == (shard_size, )
        param[shard_offset:shard_offset + shard_size] = loaded_weight


class RowParallelLinear(LinearBase):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the NPUs and we do not split
                           again.
        skip_bias_add: This was added to enable performance optimization where
                       bias can be fused with other element-wise operations.
                       We skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: Optional[bool] = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[MSDtype] = None,
        reduce_results: Optional[bool] = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: Optional[bool] = True,
    ):
        super().__init__(input_size,
                         output_size,
                         skip_bias_add,
                         params_dtype,
                         quant_config,
                         prefix,
                         return_bias=return_bias)

        # Divide the weight matrix along the last dimension.
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        assert self.quant_method is not None

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2 if self.quant_method.__class__.__name__
                in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader),
        )
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError("When not reduce the results, adding bias to the "
                             "results can lead to incorrect results")

        if bias:
            self.bias = Parameter(mint.zeros(self.output_size,
                                             dtype=self.params_dtype),
                                  requires_grad=False)
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.bias = None

        self.tensor_model_parallel_all_reduce = ReduceFromModelParallelRegion()

    def construct(self, input_):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self,
                                                  input_parallel,
                                                  bias=bias_)
        if self.reduce_results and self.tp_size > 1:
            output = self.tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias

    def weight_loader(self, param, loaded_weight):
        tp_rank = get_tensor_model_parallel_rank()
        input_dim = getattr(param, "input_dim", None)
        shard_size = self.input_size_per_partition
        start_idx = tp_rank * shard_size
        loaded_weight = split_loaded_weight(loaded_weight, input_dim,
                                            start_idx, shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param.shape == loaded_weight.shape
        param.set_data(ms.from_numpy(loaded_weight))
