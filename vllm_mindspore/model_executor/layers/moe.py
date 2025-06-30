#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
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
# ============================================================================

import copy
from typing import Optional, List

import mindspore as ms
from mindspore import Tensor, nn, Parameter, mint, ops
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer
from mindspore.communication import get_rank, get_group_size
from mindspore.common import dtype as mstype

from vllm.distributed import divide, get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

from vllm.model_executor.utils import set_weight_attrs

from vllm_mindspore.utils import STR_DTYPE_TO_MS_DTYPE
from vllm_mindspore.model_executor.layers.activation import SwiGLU
from vllm_mindspore.model_executor.layers.linear import RowParallelLinear, MergedColumnParallelLinear, LinearBase

from vllm_mindspore.model_executor.layers.linear import UnquantizedLinearMethod
from vllm_mindspore.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm_mindspore.distributed.communication_op import ReduceFromModelParallelRegion


class SharedParallelMLP(nn.Cell):
    def __init__(self, config, parallel_config, prefix: str = "",
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__(config)
        self.rank_id = ms.communication.get_rank()
        self.rank = ms.communication.get_rank()
        self.mstype = STR_DTYPE_TO_MS_DTYPE.get(config.torch_dtype)

        self.has_bias = False
        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = config.shared_expert_intermediate_size
        tp_group_size = parallel_config.tensor_parallel_size
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.ffn_hidden_size] * 2,
            bias=self.has_bias,
            prefix=f"{prefix}.gate_up_proj",
            params_dtype=self.mstype,
            quant_config=quant_config
        )

        self.act_func = SwiGLU()

        self.down_proj = RowParallelLinear(
            self.ffn_hidden_size,
            self.hidden_size,
            bias=self.has_bias,
            reduce_results=False,
            prefix=f"{prefix}.down_proj",
            params_dtype=self.mstype,
            quant_config=quant_config
        )

        self.mul = ops.Mul()
        self.reshape = ops.Reshape()
        self.rank = ms.communication.get_rank()
        self.dump = ms.ops.TensorDump()

    def construct(self, x):
        """ Construct function of mlp block. """
        gate_up_out, _ = self.gate_up_proj(x)
        hidden = self.act_func(gate_up_out)
        output, _ = self.down_proj(hidden)
        return output


class ColumnParallelGroupLinear(LinearBase):
    def __init__(
            self,
            input_size,
            output_size,
            parallel_config,
            weight_init="normal",
            bias_init="zeros",
            bias=True,
            gather_output=False,
            stride=1,
            keep_master_weight_for_test=False,
            skip_bias_add=False,
            embedding_activation_buffer=None,
            grad_output_buffer=None,
            is_expert=False,
            tp_comm_buffer_name=None,
            disable_grad_reduce=False,
            transpose_b=True,
            params_dtype=mstype.float32,
            compute_dtype=mstype.float16,
            expert_num=1,
            quant_config: Optional[QuantizationConfig] = None,
            prefix=None,
    ):
        super(ColumnParallelGroupLinear, self).__init__(input_size, output_size)

        self.input_size = input_size
        self.output_size = output_size
        self.has_bias = bias
        self.gather_output = gather_output

        self.tensor_parallel_group_size = parallel_config.tensor_parallel_size
        self.ep_size_per_partition = divide(expert_num, self.tensor_parallel_group_size)

        self.global_rank_id = get_rank()
        self.moe_ep_rank_id = self.global_rank_id

        self.expert_num = expert_num
        self.compute_dtype = compute_dtype
        self.params_dtype = params_dtype

        self.transpose_b = transpose_b if self.expert_num <= 1 else False

        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = UnquantizedLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)

        self.skip_bias_add = skip_bias_add
        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=[self.output_size],
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            is_group_mm=True,
            expert_num_per_partition=self.ep_size_per_partition,
            is_2d_smooth_scale=True,
            weight_loader=(
                self.full_weight_loader
                if self.quant_method.__class__.__name__ is "A8W8DYNLinearMethod"
                else self.weight_loader
            ),
        )

        if bias:
            bias_shape = (self.ep_size_per_partition, self.output_size)
            self.bias = Parameter(initializer(bias_init, bias_shape, params_dtype), name="bias")
            self.bias_add = P.Add()
            set_weight_attrs(
                self.bias,
                {
                    "ep_dim": 0,
                    "output_dim": 2,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.bias = None

        self.cast = P.Cast()
        self.shape = P.Shape()
        self.reshape = P.Reshape()

    def construct(self, input_parallel, group_list=None, cumsum_flag=False):
        bias = self.bias if not self.skip_bias_add else None

        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self, input_parallel, bias, group_list=group_list, cumsum_flag=cumsum_flag)
        if self.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output

    def weight_loader(self, param, loaded_weight, expert_idx):
        ep_dim =  getattr(param, "ep_dim", None)
        assert ep_dim is not None

        shard_size = param.shape[ep_dim]
        start_idx = self.moe_ep_rank_id * shard_size
        if expert_idx < start_idx or expert_idx >= (self.moe_ep_rank_id+1) * shard_size:
            return

        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param[expert_idx - start_idx].shape == loaded_weight.shape
        param[expert_idx - start_idx, :, :] = loaded_weight

    def full_weight_loader(self, param, loaded_weight):
        ep_dim =  getattr(param, "ep_dim", None)

        if ep_dim is None:
            shard_size = param.shape[ep_dim]
            start_idx = self.moe_ep_rank_id * shard_size
            loaded_weight = loaded_weight.narrow(ep_dim, start_idx, shard_size).contiguous()

        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param.shape == loaded_weight.shape
        param.set_data(loaded_weight.contiguous())


class MergedColumnParallelGroupLinear(ColumnParallelGroupLinear):
    def __init__(
            self,
            input_size,
            output_sizes: List[int],
            parallel_config,
            bias=False,
            is_expert=False,
            params_dtype=mstype.float32,
            compute_dtype=mstype.float16,
            expert_num=1,
            weight_init="normal",
            bias_init="zeros",
            **kwargs
    ):
        super(MergedColumnParallelGroupLinear, self).__init__(
            input_size=input_size,
            output_size=sum(output_sizes),
            parallel_config=parallel_config,
            bias=bias,
            is_expert=is_expert,
            expert_num=expert_num,
            params_dtype=params_dtype,
            compute_dtype=compute_dtype,
            weight_init=weight_init,
            bias_init=bias_init,
            **kwargs
        )
        self.output_sizes = output_sizes
        self.moe_tp_size = 1
        assert all(output_size % self.moe_tp_size == 0 for output_size in output_sizes)

    def weight_loader(self, param, loaded_weight, expert_idx, loaded_shard_id: Optional[int] = None):
        param_name = param.name
        ep_dim =  getattr(param, "ep_dim", None)
        assert ep_dim is not None
        output_dim = getattr(param, "output_dim", None)
        shard_size = param.shape[ep_dim]
        ep_start_idx = self.moe_ep_rank_id * shard_size
        if expert_idx < ep_start_idx or expert_idx >= (self.moe_ep_rank_id+1) * shard_size:
            return
        if ep_dim is not None and output_dim is not None:
            param_data = param.data

            self.moe_tp_size = 1
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.moe_tp_size
            shard_size = self.output_sizes[loaded_shard_id] // self.moe_tp_size

            if param_name.endswith("weight"):
                loaded_weight = loaded_weight.transpose(0,1)
                loaded_weight = loaded_weight.narrow(output_dim-1, 0, shard_size).contiguous()
            if param_name.endswith("weight_scale"):
                loaded_weight = loaded_weight.squeeze(-1)
            if  len(param.shape) == 3:
                if self.quant_method.gmm_transpose:
                    param[expert_idx - ep_start_idx, shard_offset : shard_offset + shard_size, :] = loaded_weight.transpose().contiguous()
                else:
                    param[expert_idx - ep_start_idx, :, shard_offset : shard_offset + shard_size] = loaded_weight
            else:
                param[expert_idx - ep_start_idx, shard_offset : shard_offset + shard_size] = loaded_weight
            return
        else:
            if len(param[expert_idx - ep_start_idx].shape) == 0 and loaded_weight.shape == (1,):
                param_data = param.asnumpy()
                param_data[expert_idx - ep_start_idx] = loaded_weight[0].asnumpy()
                param.data.copy_(ms.tensor(param_data))
            else:
                assert param[expert_idx - ep_start_idx].shape == loaded_weight.shape
                param[expert_idx - ep_start_idx] = loaded_weight

    def full_weight_loader(self, param, loaded_weight, loaded_shared_id: Optional[int] = None):
        output_dim = getattr(param, "output_dim", None)
        input_dim = getattr(param, "input_dim", None)
        ep_dim =  getattr(param, "ep_dim", None)
        param_name = param.name
        if ep_dim is not None and output_dim is not None:
            param_data = param.data
            ep_shard_size = param.shape[ep_dim]
            ep_start_idx = self.moe_ep_rank_id * ep_shard_size
            loaded_weight = loaded_weight.narrow(ep_dim, ep_start_idx, ep_shard_size).contiguous()

            self.moe_tp_size = 1
            shard_offset = sum(self.output_sizes[:loaded_shared_id]) // self.moe_tp_size
            shard_size = self.output_sizes[loaded_shared_id] // self.moe_tp_size

            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            if len(param.shape) == 3:
                param[:, :, shard_offset : shard_offset + shard_size] = loaded_weight
            else:
                param[:, shard_offset : shard_offset + shard_size] = loaded_weight
            return
        if input_dim is not None:
            param_data = param.data
            assert param.shape == loaded_weight.shape
            param_data.copy_(loaded_weight)
            return

        assert param.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class RowParallelGroupLinear(LinearBase):
    def __init__(
            self,
            input_size,
            output_size,
            parallel_config,
            input_is_parallel,
            weight_init="normal",
            bias_init="zeros",
            bias=True,
            skip_bias_add=False,
            stride=1,
            keep_master_weight_for_test=False,
            is_expert=False,
            tp_comm_buffer_name=None,
            transpose_b=True,
            params_dtype=mstype.float32,
            compute_dtype=mstype.float16,
            expert_num=1,
            delay_allreduce=False,
            prefix=None,
            quant_config: Optional[QuantizationConfig] = None,
    ):
        super(RowParallelGroupLinear, self).__init__(input_size, output_size)

        self.input_size = input_size
        self.output_size = output_size
        self.has_bias = bias
        self.skip_bias_add = skip_bias_add
        self.input_is_parallel = input_is_parallel

        self.compute_dtype = compute_dtype
        self.expert_num = expert_num
        self.is_expert = is_expert
        self.transpose_b = transpose_b if self.expert_num <= 1 else False
        self.delay_allreduce = delay_allreduce

        self.tensor_parallel_group_size = parallel_config.tensor_parallel_size
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.params_dtype = params_dtype

        self.ep_size_per_partition = divide(expert_num, self.tensor_parallel_group_size)

        self.moe_tp_size = 1
        self.global_rank_id = get_rank()
        self.moe_ep_rank_id = self.global_rank_id // self.moe_tp_size
        self.moe_tp_rank_id = self.global_rank_id % self.moe_tp_size

        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = UnquantizedLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=[self.output_size],
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            is_group_mm=True,
            expert_num_per_partition=self.ep_size_per_partition,
            weight_loader=(
                self.full_weight_loader
                if self.quant_method.__class__.__name__ is "A8W8DYNLinearMethod"
                else self.weight_loader
            ),
        )
        self.tensor_model_parallel_all_reduce = ReduceFromModelParallelRegion()

        if self.has_bias and not self.skip_bias_add:
            bias_shape = (self.ep_size_per_partition, self.output_size)
            self.bias = Parameter(initializer(super().bias_init, bias_shape, super().params_dtype), name="bias")
            self.bias_add = P.Add()
            set_weight_attrs(
                self.bias,
                {
                    "ep_dim": 0,
                    "output_dim": 1,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.bias = None

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()

    def construct(self, input_, group_list=None, cumsum_flag=False):
        """
        Forward of RowParallelLinear.
        Performs a linear transformation considering various parallel modes and data type conversions.
        """
        input_parallel = input_
        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self, input_parallel, bias=bias_,  group_list=group_list, cumsum_flag=cumsum_flag)

        return output_parallel

    def weight_loader(self, param, loaded_weight, expert_idx):
        # input_dim = getattr(param, "input_dim", None)
        ep_dim =  getattr(param, "ep_dim", None)
        assert ep_dim is not None
        param_name = param.name
        shard_size = param.shape[ep_dim]
        start_idx = self.moe_ep_rank_id * shard_size
        if expert_idx < start_idx or expert_idx >= (self.moe_ep_rank_id+1) * shard_size:
            return
        if param_name.endswith("weight"):
            loaded_weight = loaded_weight.transpose(0, 1)
        if param_name.endswith("weight_scale"):
            loaded_weight = loaded_weight.squeeze(-1)
        if len(param[expert_idx - start_idx].shape) == 0 and loaded_weight.shape == (1,):
            param_data = param.asnumpy()
            param_data[expert_idx - start_idx] = loaded_weight[0].asnumpy()
            param.data.copy_(ms.tensor(param_data))
        else:
            if self.quant_method.gmm_transpose:
                loaded_weight = loaded_weight.transpose().contiguous()
            assert param[expert_idx - start_idx].shape == loaded_weight.shape
            param[expert_idx - start_idx] = loaded_weight


    def full_weight_loader(self, param, loaded_weight):
        ep_dim =  getattr(param, "ep_dim", None)
        param_data = param.data
        if ep_dim is not None:
            shard_size = param.shape[ep_dim]
            start_idx = self.moe_ep_rank_id * shard_size
            loaded_weight = loaded_weight.narrow(ep_dim, start_idx, shard_size).contiguous()
        self.moe_tp_rank_id  = get_rank()

        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class RoutedParallelMLP(nn.Cell):
    def __init__(self, config, parallel_config, prefix: Optional[str] = None,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = config.moe_intermediate_size
        self.act_func = SwiGLU()
        self.mstype = STR_DTYPE_TO_MS_DTYPE.get(config.torch_dtype)

        self.gate_up_proj = MergedColumnParallelGroupLinear(
            self.hidden_size,
            [self.ffn_hidden_size] * 2,
            parallel_config=parallel_config,
            bias=False,
            is_expert=True,
            transpose_b=True,
            expert_num=config.num_experts,
            params_dtype=self.mstype,
            compute_dtype=self.mstype,
            prefix=f'{prefix}.gate_up_proj',
            quant_config=quant_config
        )

        self.down_proj = RowParallelGroupLinear(
            self.ffn_hidden_size,
            self.hidden_size,
            input_is_parallel=True,
            parallel_config=parallel_config,
            bias=False,
            skip_bias_add=True,
            transpose_b=True,
            params_dtype=self.mstype,
            compute_dtype=self.mstype,
            is_expert=True,
            expert_num=config.num_experts,
            delay_allreduce=True,
            quant_config=quant_config,
            prefix=f'{prefix}.down_proj'
        )

    def construct(self, x, group_list=None, cumsum_flag=False):
        gate_hidden_out = self.gate_up_proj(x, group_list=group_list, cumsum_flag=cumsum_flag)
        hidden = self.act_func(gate_hidden_out)
        output = self.down_proj(hidden, group_list, cumsum_flag=cumsum_flag)
        return output
