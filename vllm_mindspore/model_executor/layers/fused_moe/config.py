# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/config.py
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

from dataclasses import dataclass
from enum import Enum

import mindspore as ms
import vllm.envs as envs
from vllm.config import ParallelConfig, get_current_vllm_config
from vllm.distributed import (get_dp_group, get_ep_group,
                              get_tensor_model_parallel_rank)
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class FusedMoEParallelConfig:
    tp_size: int
    dp_size: int
    ep_size: int
    tp_rank: int
    dp_rank: int
    ep_rank: int

    use_ep: bool  # whether to use EP or not

    @property
    def use_all2all_kernels(self):
        return self.dp_size > 1 and self.use_ep and self.tp_size == 1

    @staticmethod
    def make(tp_size_: int, dp_size_: int,
             vllm_parallel_config: ParallelConfig) -> "FusedMoEParallelConfig":
        """
        Determine MoE parallel configuration. Based on the input tp_size_,
        dp_size_, ep_size_ and vllm's parallel config, determine what
        level's of parallelism to use in the fused moe layer.

        Args:
            tp_size_ (int): tp_size passed into the FusedMoE constructor.
            dp_size_ (int): dp_size passed into the FusedMoE constructor.
            ep_size_ (int): ep_size passed into the FusedMoE constructor.
            vllm_parallel_config (ParallelConfig): vllm's parallel config
            object.

        Examples:
        When there is no parallelism requested, i.e. tp_size_ = dp_size_ = 1,
        we simply return the sizes unaltered and the ranks set to 0.

        Expert Parallelism is considered only when either dp_size_ or tp_size_
        is non trivial.

        When TP = 2, DP = 1 and EP = False, the configuration on different
        devices,
            - device 0 : TP = {2, 0} DP = {1, 0} EP = {1, 0} //
                         legend : {size, rank}
            - device 1 : TP = {2, 1} DP = {1, 0} EP = {1, 0}
            - Comment : Tensors are sharded across 2 devices.

        When TP = 1, DP = 2 and EP = False, the configuration on different
        devices,
            - device 0 : TP = {2, 0} DP = {2, 0} EP = {1, 0}
            - device 1 : TP = {2, 1} DP = {2, 1} EP = {1, 0}
            - Comment: There are 2 engine instances and the tensors are sharded
              across 2 decvices.

        When TP = 2, DP = 2 and EP = False, the configuration on different
        devices,
            - device 0: TP = {4, 0} DP = {2, 0} EP = {1, 0}
            - device 1: TP = {4, 1} DP = {2, 0} EP = {1, 0}
            - device 2: TP = {4, 2} DP = {2, 1} EP = {1, 0}
            - device 3: TP = {4, 3} DP = {2, 1} EP = {1, 0}
            - Comment: There are 2 engine instances and the tensors are sharded
              across 4 devices.

        When, TP = 2, DP = 1 and EP = True, the configuration on different
        devices,
            - device 0: TP = {1, 0} DP = {1, 0} EP = {2, 0}
            - device 1: TP = {1, 0} DP = {1, 0} EP = {2, 1}
            - Comment: The experts are split between the 2 devices.

        When, TP = 1, DP = 2 and EP = True, the configuration on different
        devices,
            - device 0: TP = {1, 0} DP = {2, 0} EP = {2, 0}
            - device 1: TP = {1, 0} DP = {2, 1} EP = {2, 1}
            - Comment: There are 2 engine instances and the experts are split
              between the 2 devices.

        When TP = 2, DP = 2 and EP = True, the configuration on different
        devices,
            - device 0: TP = {1, 0} DP = {2, 0} EP = {4, 0}
            - device 1: TP = {1, 0} DP = {2, 0} EP = {4, 1}
            - device 2: TP = {1, 0} DP = {2, 1} EP = {4, 2}
            - device 3: TP = {1, 0} DP = {2, 1} EP = {4, 3}
            - Comment: There are 2 engine instances and the experts are split
              between the 4 devices.

        When TP = 2, DP = 2, EP = True and expert_parallel = 2,
        the configuration on different devices,
            - device 0: TP = {2, 0} DP = {2, 0} EP = {2, 0}
            - device 1: TP = {2, 1} DP = {2, 0} EP = {2, 0}
            - device 2: TP = {2, 0} DP = {2, 1} EP = {2, 1}
            - device 3: TP = {2, 1} DP = {2, 1} EP = {2, 1}
            - Comment: There are 2 engine instances and the experts are split
              between the 2 devices.
        """

        def flatten_tp_across_dp(dp_rank: int):
            tp_rank = 0 if tp_size_ == 1 else get_tensor_model_parallel_rank()
            # There are actually dp_size_ * tp_size_ devices. Update tp_size
            # and tp_rank so we shard across all devices.
            tp_size = dp_size_ * tp_size_
            tp_rank = dp_rank * tp_size_ + tp_rank
            return tp_size, tp_rank

        use_ep = (dp_size_ * tp_size_ > 1
                  and vllm_parallel_config.enable_expert_parallel)

        dp_size = dp_size_
        dp_rank = get_dp_group().rank_in_group if dp_size > 1 else 0
        tp_size, tp_rank = flatten_tp_across_dp(dp_rank)

        if not use_ep:
            return FusedMoEParallelConfig(tp_size=tp_size,
                                          tp_rank=tp_rank,
                                          dp_size=dp_size,
                                          dp_rank=dp_rank,
                                          ep_size=1,
                                          ep_rank=0,
                                          use_ep=False)
        # DP + EP / TP + EP / DP + TP + EP
        assert use_ep

        vllm_config = get_current_vllm_config()
        # custom_ep_size is used for tp + ep parallel,
        # which is not supported in original vllm.
        if vllm_config.additional_config is not None and \
           vllm_config.additional_config.get("expert_parallel", None) \
           is not None:
            custom_ep_size = int(
                vllm_config.additional_config.get("expert_parallel", None))
            ep_size = custom_ep_size
            tp_size = tp_size // custom_ep_size
            tp_rank = tp_rank % tp_size
            ep_rank = get_ep_group().rank_in_group // tp_size
            return FusedMoEParallelConfig(tp_size=tp_size,
                                          tp_rank=tp_rank,
                                          dp_size=dp_size,
                                          dp_rank=dp_rank,
                                          ep_size=ep_size,
                                          ep_rank=ep_rank,
                                          use_ep=True)
        else:
            # In EP, each device owns a set of experts fully.
            # There is no tensor parallel update tp_size, tp_rank,
            # ep_size and ep_rank to reflect that.
            ep_size = tp_size
            ep_rank = tp_rank
            return FusedMoEParallelConfig(tp_size=1,
                                          tp_rank=0,
                                          dp_size=dp_size,
                                          dp_rank=dp_rank,
                                          ep_size=ep_size,
                                          ep_rank=ep_rank,
                                          use_ep=True)


@dataclass
class FusedMoEConfig:
    num_experts: int
    experts_per_token: int
    hidden_dim: int

    num_local_experts: int
    moe_parallel_config: FusedMoEParallelConfig

    in_dtype: ms.dtype.Type  # The activation type.
    quant_dtype: ms.dtype.Type = None

    # TODO: add more quantization params, blocked, per-token, etc.
    block_size: int = 128

    max_num_tokens: int = envs.VLLM_FUSED_MOE_CHUNK_SIZE

    def __post_init__(self):
        if self.dp_size > 1:
            logger.debug("Using FusedMoEConfig::max_num_tokens=%d",
                         self.max_num_tokens)

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


class MoeMode(Enum):
    PURE_TP = 0
    PURE_EP = 1
    TP_MIX_EP = 2