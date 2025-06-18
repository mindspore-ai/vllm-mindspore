# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/layer.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2025 The vLLM team.
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
"""Fused MoE layers with MindSpore."""

from abc import abstractmethod
from typing import Callable, Optional

import numpy as np
import vllm.envs as envs
from mindspore import Parameter, Tensor, from_numpy, mint, nn, ops
from vllm.config import get_current_vllm_config
from vllm.distributed import (get_dp_group, get_ep_group,
                              get_tensor_model_parallel_world_size,
                              get_tp_group)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs

from vllm_mindspore.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEParallelConfig, MoeMode)
from vllm_mindspore.model_executor.layers.fused_moe.fused_moe import (
    FusedExperts, fused_topk, grouped_topk)
from vllm_mindspore.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase)
from vllm_mindspore.model_executor.model_loader.weight_utils import (
    split_loaded_weight)

logger = init_logger(__name__)


class FusedMoEMethodBase(QuantizeMethodBase):

    @abstractmethod
    def create_weights(self, layer: nn.Cell, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype, **extra_weight_attrs):
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: nn.Cell,
        x: Tensor,
        router_logits: Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> Tensor:
        raise NotImplementedError


class UnquantizedFusedMoEMethod(FusedMoEMethodBase, nn.Cell):
    """MoE method without quantization."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__()
        self.fused_experts = FusedExperts(moe)
        self.moe = moe

    def create_weights(self, layer: nn.Cell, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype, **extra_weight_attrs):
        # Fused gate_up_proj (column parallel)
        # Transpose the weight to make it compatible with the GroupMatMul kernel
        w13_weight = Parameter(mint.empty(num_experts,
                                          hidden_size,
                                          2 * intermediate_size_per_partition,
                                          dtype=params_dtype),
                               requires_grad=False)
        layer.insert_param_to_cell("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        # Mark the weight as transposed, so the weight loader can know it.
        set_weight_attrs(w13_weight, {"is_transposed": True})

        # down_proj (row parallel)
        w2_weight = Parameter(mint.empty(num_experts,
                                         intermediate_size_per_partition,
                                         hidden_size,
                                         dtype=params_dtype),
                              requires_grad=False)
        layer.insert_param_to_cell("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        set_weight_attrs(w2_weight, {"is_transposed": True})

    def apply(
        self,
        layer: nn.Cell,
        x: Tensor,
        router_logits: Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> Tensor:
        return self.forward_npu(
            x=x,
            layer=layer,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input)

    def forward_npu(
        self,
        layer: nn.Cell,
        x: Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> Tensor:
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=None)

        return self.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=apply_router_weight_on_input)


def determine_expert_map(ep_size: int, ep_rank: int, global_num_experts: int):
    """
    use numpy rather than tensor because tensor operation will be on NPU
    with mindspore, which is slow.
    """
    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None)

    local_num_experts = global_num_experts // ep_size

    # Create a numpy array of size global_num_experts filled with -1
    expert_map = np.full((global_num_experts, ), -1, dtype=np.int32)
    # Create an expert map for the local experts
    if ep_rank < (ep_size - 1):
        # Each non-last rank gets local_num_experts experts.
        expert_map[ep_rank * local_num_experts:
                   (ep_rank + 1) * local_num_experts] = \
            np.arange(0, local_num_experts, dtype=np.int32)
    else:
        # All remaining experts are assigned to the last rank.
        local_num_experts = global_num_experts - ep_rank * local_num_experts
        expert_map[-local_num_experts:] = np.arange(0,
                                                    local_num_experts,
                                                    dtype=np.int32)
    return (local_num_experts, expert_map)


class FusedMoE(nn.Cell):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all all_reduce on the output of the layer
        renomalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
    """

    def __init__(
            self,
            num_experts: int,  # Global number of experts
            top_k: int,
            hidden_size: int,
            intermediate_size: int,
            params_dtype=None,
            reduce_results: bool = False,
            renormalize: bool = True,
            use_grouped_topk: bool = False,
            num_expert_group: Optional[int] = None,
            topk_group: Optional[int] = None,
            quant_config: Optional[QuantizationConfig] = None,
            tp_size: Optional[int] = None,
            ep_size: Optional[int] = None,
            dp_size: Optional[int] = None,
            prefix: str = "",
            custom_routing_function: Optional[Callable] = None,
            scoring_func: str = "softmax",
            e_score_correction_bias: Optional[Tensor] = None,
            apply_router_weight_on_input: bool = False,
            activation: str = "silu"):
        super().__init__()

        # TODO： to support apply_router_weight_on_input
        if apply_router_weight_on_input:
            raise NotImplementedError("apply_router_weight_on_input"
                                      "is not supported yet")

        if params_dtype is None:
            params_dtype = get_current_vllm_config().model_config.dtype
        self.params_dtype = params_dtype

        vllm_config = get_current_vllm_config()
        self.moe_parallel_config: FusedMoEParallelConfig = (
            FusedMoEParallelConfig.make(
                tp_size_=(tp_size if tp_size is not None else
                          get_tensor_model_parallel_world_size()),
                dp_size_=(dp_size if dp_size is not None else
                          get_dp_group().world_size),
                vllm_parallel_config=vllm_config.parallel_config))

        self.global_num_experts = num_experts

        # Determine expert maps
        if self.use_ep:
            self.local_num_experts, self.expert_map = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts)
        else:
            self.local_num_experts, self.expert_map = (self.global_num_experts,
                                                       None)

        # Determine the moe parallel mode.
        # pure_tp means using tensor parallelism only, no expert parallelism.
        self.fused_moe_mode = None

        # self.ep_size == 1, means use tensor parallelism to compute moe.
        if self.ep_size == 1:
            self.fused_moe_mode = MoeMode.PURE_TP
        # self.ep_size > 1, means use expert parallelism or
        # expert parallelism mix tensor parallelism.
        else:
            if self.tp_size == 1:
                self.fused_moe_mode = MoeMode.PURE_EP
            else:
                self.fused_moe_mode = MoeMode.TP_MIX_EP

        if self.ep_rank < (self.ep_size - 1):
            self.expert_start_index = self.ep_rank * self.local_num_experts
            self.expert_end_index = (self.ep_rank + 1) * self.local_num_experts
        else:
            self.expert_start_index = self.ep_rank * self.local_num_experts
            self.expert_end_index = self.global_num_experts

        self.top_k = top_k

        assert intermediate_size % self.tp_size == 0
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.e_score_correction_bias = e_score_correction_bias
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.activation = activation

        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError("Only softmax scoring function is supported for "
                             "non-grouped topk.")

        moe = FusedMoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            num_local_experts=self.local_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            # TODO (bnell): this needs to be fixed for quantized types.
            in_dtype=params_dtype,
            max_num_tokens=envs.VLLM_FUSED_MOE_CHUNK_SIZE,
        )
        self.moe_config = moe
        self.quant_config = quant_config

        # Note: get_quant_method will look at the layer's local_num_experts
        # for heuristic purposes, so it must be initialized first.
        quant_method: Optional[QuantizeMethodBase] = None

        if quant_config is None:
            quant_method = UnquantizedFusedMoEMethod(moe)
        else:
            quant_method = quant_config.get_quant_method(self, prefix)

        assert quant_method is not None
        assert isinstance(quant_method, FusedMoEMethodBase)
        self.quant_method = quant_method

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition":
            self.intermediate_size_per_partition,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if (self.quant_method.__class__.__name__
                in ("GPTQMarlinMoEMethod",
                    "CompressedTensorsWNA16MarlinMoEMethod",
                    "CompressedTensorsWNA16MoEMethod")):
            moe_quant_params["intermediate_size_full"] = intermediate_size

        self.quant_method.create_weights(layer=self, **moe_quant_params)

        # Initialize some communication ops and group.
        self.dp_group = get_dp_group().device_group._name
        self.ep_group = get_ep_group().device_group._name

        self.tp_world_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group().device_group._name
        self.all_reduce_from_tp_group = ops.AllReduce(group=self.tp_group)

        if (self.pure_tp or self.tp_ep) and self.dp_size > 1:
            self.all_gather_from_dp_group = ops.AllGather(group=self.dp_group)
            self.all_reduce_from_dp_group = ops.AllReduce(group=self.dp_group)
            self.reduce_scatter_from_dp_group = ops.ReduceScatter(
                group=self.dp_group)

    @property
    def pure_tp(self):
        return self.fused_moe_mode == MoeMode.PURE_TP

    @property
    def pure_ep(self):
        return self.fused_moe_mode == MoeMode.PURE_EP

    @property
    def tp_ep(self):
        return self.fused_moe_mode == MoeMode.TP_MIX_EP

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def dp_size(self):
        return self.moe_parallel_config.dp_size

    @property
    def ep_size(self):
        return self.moe_parallel_config.ep_size

    @property
    def tp_rank(self):
        return self.moe_parallel_config.tp_rank

    @property
    def dp_rank(self):
        return self.moe_parallel_config.dp_rank

    @property
    def ep_rank(self):
        return self.moe_parallel_config.ep_rank

    @property
    def use_ep(self):
        return self.moe_parallel_config.use_ep

    @property
    def use_all2all_kernels(self):
        return self.moe_parallel_config.use_all2all_kernels

    def _load_w13(self, param: Parameter, shard_dim: int, shard_id: str,
                  loaded_weight: Tensor, expert_id: int, tp_rank: int):
        is_param_transpose = param.is_transposed \
            if hasattr(param, "is_transposed") else False

        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        if is_param_transpose:
            shard_size = param.shape[-1] // 2
        else:
            shard_size = param.shape[-2] // 2

        loaded_weight = split_loaded_weight(loaded_weight, shard_dim,
                                            shard_size * tp_rank, shard_size)

        if is_param_transpose:
            loaded_weight = from_numpy(loaded_weight.swapaxes(-1, -2))
        else:
            loaded_weight = from_numpy(loaded_weight)

        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            if is_param_transpose:
                param[expert_id, :, 0:shard_size] = loaded_weight
            else:
                param[expert_id, 0:shard_size, :] = loaded_weight
        # w3, up_proj: Load into second logical weight of w13.
        else:
            assert shard_id == "w3"
            if is_param_transpose:
                param[expert_id, :, shard_size:shard_size * 2] = loaded_weight
            else:
                param[expert_id, shard_size:shard_size * 2, :] = loaded_weight

    def _load_w2(self,
                 param: Parameter,
                 shard_dim: int,
                 loaded_weight: Tensor,
                 tp_rank: int,
                 expert_id: int,
                 load_full: bool = False):
        is_param_transpose = param.is_transposed \
            if hasattr(param, "is_transposed") else False

        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        if not load_full:
            if is_param_transpose:
                shard_size = param.shape[-2]
            else:
                shard_size = param.shape[-1]
            loaded_weight = split_loaded_weight(loaded_weight, shard_dim,
                                                shard_size * tp_rank,
                                                shard_size)

            if is_param_transpose:
                loaded_weight = from_numpy(loaded_weight.swapaxes(-1, -2))
            else:
                loaded_weight = from_numpy(loaded_weight)
            param[expert_id] = loaded_weight
        # w2, down_proj: Load into only logical weight of w2.
        else:
            if is_param_transpose:
                loaded_weight = from_numpy(loaded_weight.swapaxes(-1, -2))
            else:
                loaded_weight = from_numpy(loaded_weight)
            param.set_data(loaded_weight)

    def _load_single_value(self, param: Parameter, loaded_weight: Tensor,
                           expert_id: int):
        is_param_transpose = param.is_transposed \
            if hasattr(param, "is_transposed") else False
        loaded_weight = loaded_weight[:]
        if is_param_transpose:
            loaded_weight = from_numpy(loaded_weight.swapaxes(-1, -2))
        else:
            loaded_weight = from_numpy(loaded_weight)
        param[expert_id] = from_numpy(loaded_weight)

    def _load_g_idx(self, shard_id: str, param: Parameter, shard_dim: int,
                    loaded_weight: Tensor, tp_rank: int, expert_id: int):

        if shard_id == "w2":
            self._load_w2(shard_dim=shard_dim,
                          loaded_weight=loaded_weight,
                          param=param,
                          expert_id=expert_id,
                          tp_rank=tp_rank)
        else:
            assert shard_id in ("w1", "w3")
            is_param_transpose = param.is_transposed \
                if hasattr(param, "is_transposed") else False
            loaded_weight = loaded_weight[:]
            if is_param_transpose:
                loaded_weight = from_numpy(loaded_weight.swapaxes(-1, -2))
            else:
                loaded_weight = from_numpy(loaded_weight)
            param[expert_id] = from_numpy(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        if self.expert_map is None:
            return expert_id
        return self.expert_map[expert_id].item()

    def _load_model_weight_or_group_weight_scale(self,
                                                 shard_dim: int,
                                                 param: Parameter,
                                                 shard_id: str,
                                                 loaded_weight: Tensor,
                                                 tp_rank: int,
                                                 expert_id: int,
                                                 load_full_w2: bool = False):
        """
        Load grouped weight scales for group quantization or model weights
            :param shard_dim: dimension to shard
            :param expert_data: parameter for a particular expert
            :param shard_id: either w1, w2, or w3
            :param loaded_weight: checkpoint weight to load into the param
            :param tp_rank: tensor parallel rank
            :param load_full_w2: whether or not the w2 loaded should be sharded.
        """
        if shard_id == "w2":
            # In the case where we have actorder/g_idx, we do not partition the
            # w2 scales, as indicated by `load_full` argument, for all tp cases
            self._load_w2(shard_dim=shard_dim,
                          loaded_weight=loaded_weight,
                          param=param,
                          tp_rank=tp_rank,
                          expert_id=expert_id,
                          load_full=load_full_w2)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           param=param,
                           expert_id=expert_id,
                           tp_rank=tp_rank)

    def weight_loader(self, param: Parameter, loaded_weight: Tensor,
                      weight_name: str, shard_id: str, expert_id: int) -> None:

        expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
        if expert_id == -1:
            return

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but "
                             f"got {shard_id}.")

        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]

        # TODO: full_load will slow down the loading process,
        # Support it when need it in the future.

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(shard_dim=0,
                             shard_id=shard_id,
                             loaded_weight=loaded_weight,
                             param=param,
                             tp_rank=self.tp_rank,
                             expert_id=expert_id)
            return

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(param=param,
                                    loaded_weight=loaded_weight,
                                    expert_id=expert_id)
            return

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                param=param,
                expert_id=expert_id,
                tp_rank=self.tp_rank)
            return

    @staticmethod
    def select_experts(hidden_states: Tensor,
                       router_logits: Tensor,
                       top_k: int,
                       use_grouped_topk: bool,
                       renormalize: bool,
                       topk_group: Optional[int] = None,
                       num_expert_group: Optional[int] = None,
                       custom_routing_function: Optional[Callable] = None,
                       scoring_func: str = "softmax",
                       e_score_correction_bias: Optional[Tensor] = None,
                       indices_type=None):

        # DeekSeekv2 uses grouped_top_k
        if use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias)
            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)
        elif custom_routing_function is None:
            topk_weights, topk_ids = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                indices_type=indices_type,
            )
        else:
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)
            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)

        return topk_weights, topk_ids

    def must_reduce_shared_expert_outputs(self) -> bool:
        # If use tp moe, there is a delay all reduce ops of routed experts.
        # Therefore, shared experts in tensor parallelism can perform
        # all-reduce operations together with routing experts
        return not (self.pure_tp or self.tp_ep)

    def maybe_all_reduce_tensor_model_parallel(self,
                                               final_hidden_states: Tensor):
        """
        To all_reduce after routed expert and shared expert are added.
        """
        # Do delay allreduce If "must_reduce_shared_expert_outputs" return True
        if self.pure_tp or self.tp_ep:
            return self.all_reduce_from_tp_group(final_hidden_states)
        return final_hidden_states

    def construct(self,
                  hidden_states: Tensor,
                  router_logits: Tensor,
                  dp_pad_index=None,
                  dp_unpad_index=None,
                  dp_pad_index_with_offset=None,
                  dp_unpad_index_total_with_offset=None):
        if self.use_all2all_kernels:
            return self.forward_impl_chunked(hidden_states, router_logits)

        return self.forward_impl(hidden_states, router_logits, dp_pad_index,
                                 dp_unpad_index, dp_pad_index_with_offset,
                                 dp_unpad_index_total_with_offset)

    def forward_impl(self, hidden_states: Tensor, router_logits: Tensor,
                     dp_pad_index, dp_unpad_index,
                     dp_pad_index_total_with_offset,
                     dp_unpad_index_total_with_offset):
        """
        If dp_world_size == 4, dp_rank == 1,
          tokens_num across dp is [1, 3, 4, 2], then
            dp_pad_index = [0, 1, 2, 0]
            dp_unpad_index = [0, 1, 2]
            dp_pad_index_total_with_offset = \
               [0, 0, 0, 0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 0, 0]
            dp_unpad_index_total_with_offset = \
               [0, 4, 5, 6, 8, 9, 10, 11, 12, 13]
        """
        if (self.pure_tp or self.tp_ep) and self.dp_size > 1:
            hidden_logit_buffer = mint.cat((hidden_states, router_logits),
                                           dim=-1)
            # TODO: replace AllGather with AllGatherV to eliminate padding
            hidden_logit_buffer = mint.index_select(hidden_logit_buffer, 0,
                                                    dp_pad_index)
            hidden_logit_buffer = self.all_gather_from_dp_group(
                hidden_logit_buffer)
            hidden_logit_buffer = mint.index_select(
                hidden_logit_buffer, 0, dp_unpad_index_total_with_offset)
            hidden_states, router_logits = mint.split(
                hidden_logit_buffer,
                [hidden_states.shape[-1], router_logits.shape[-1]],
                dim=-1)

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )

        if (self.pure_tp or self.tp_ep) and self.dp_size > 1:
            # TODO: replace ReudceScatter with ReduceScatterV
            # to eliminate padding
            final_hidden_states = mint.index_select(
                final_hidden_states, 0, dp_pad_index_total_with_offset)
            final_hidden_states = self.reduce_scatter_from_dp_group(
                final_hidden_states)
            final_hidden_states = mint.index_select(final_hidden_states, 0,
                                                    dp_unpad_index)

        if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
            # Default set to False. (May have to add shared expert outputs.)
            final_hidden_states = self.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states)

        return final_hidden_states

    def forward_impl_chunked(self, full_hidden_states: Tensor,
                             full_router_logits: Tensor):
        # TODO: to implement chunked forward for FusedMoE.
        # Chunked forward can solve the batch size limitation
        # of the dispatch-combine kernel.

        hidden_states = full_hidden_states
        router_logits = full_router_logits

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )

        return final_hidden_states

    @classmethod
    def make_expert_params_mapping(
            cls, ckpt_gate_proj_name: str, ckpt_down_proj_name: str,
            ckpt_up_proj_name: str,
            num_experts: int) -> list[tuple[str, str, int, str]]:

        return [
            # the format is (param_name, weight_name, expert_id, shard_id)
            ("experts.w13_" if weight_name
             in [ckpt_gate_proj_name, ckpt_up_proj_name] else "experts.w2_",
             f"experts.{expert_id}.{weight_name}.", expert_id, shard_id)
            for expert_id in range(num_experts) for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    def extra_repr(self) -> str:

        s = (
            f"global_num_experts={self.global_num_experts}, "
            f"local_num_experts={self.local_num_experts}, "
            f"top_k={self.top_k}, "
            f"intermediate_size_per_partition={self.intermediate_size_per_partition}, "  # noqa: E501
            f"tp_size={self.tp_size},\n"
            f"ep_size={self.ep_size}, "
            f"reduce_results={self.reduce_results}, "
            f"renormalize={self.renormalize}, "
            f"use_grouped_topk={self.use_grouped_topk}")

        if self.use_grouped_topk:
            s += f", num_expert_group={self.num_expert_group}, topk_group={self.topk_group}"  # noqa: E501

        s += f", scoring_func='{self.scoring_func}', activation='{self.activation}'"  # noqa: E501

        return s
