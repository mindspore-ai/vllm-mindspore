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

from typing import Iterable, Set, Tuple

from vllm.config import VllmConfig
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger

from mindspore import Tensor, JitConfig

from mindformers.models.llama import LlamaConfig as LlamaConfig_MF
from research.qwen2_5.infer.qwen2_5 import (
    ParallelQwenForCausalLM as ParallelQwenForCausalLM_MF,
)

from vllm_mindspore.model_executor.layers.sampler import get_sampler
from vllm_mindspore.model_executor.models.mf_models.mf_model_base import MfModelBase, Fake_Attention
from vllm_mindspore.utils import calc_block_num
from vllm_mindspore.model_executor.models.mf_models.qwen2_infer_parallelism import Qwen2InferParallelism
from vllm_mindspore.model_executor.models.mf_models.attention_mask import LowerTriangularMask


logger = init_logger(__name__)


class Qwen2ForCausalLM(MfModelBase):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super(Qwen2ForCausalLM, self).__init__(vllm_config=vllm_config, prefix=prefix)

        self.mf_model_config = LlamaConfig_MF(**self.mf_config.model.model_config)
        # Cannot get num_gpu_blocks from cache config now, calculate one first.
        self.mf_model_config.num_blocks = calc_block_num(
            self.cache_config, self.model_config, self.parallel_config
        )
        self.mf_model_config.block_size = self.cache_config.block_size
        if self.mf_config.moe_config:
            self.mf_model_config.moe_config = self.mf_config.moe_config
        self.mf_model_config.return_hidden_states = True

        # qwen qkv concat will support in next version
        self.mf_model_config.qkv_concat = False
        self.mf_config.model.model_config.qkv_concat = False
        # Initial network
        self.network = ParallelQwenForCausalLM_MF(self.mf_model_config)
        self.network._jit_config_dict = JitConfig(
            jit_level="O0", infer_boost="on"
        ).jit_config_dict

        self.mf_config.load_checkpoint = self.get_model_path()

        self.mf_kvcaches_init = False

        self.sampler = get_sampler()
        self.set_modules({"model": self.network})

        self.kv_caches = [Fake_Attention() for i in range(self.mf_model_config.num_layers)]
        compilation_config = get_current_vllm_config().compilation_config

        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(self.mf_model_config.num_layers):
            compilation_config.static_forward_context[str(i)] = self.kv_caches[i]

        self.casual_mask = LowerTriangularMask(mf_model_config=self.mf_model_config)
        self.set_flags = False

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> Set[str]:
        model_parallelism = Qwen2InferParallelism(self.mf_config, self.network, False)
        model_parallelism.infer_convert_and_parallelism(self.mf_config.load_checkpoint)

        self.network.set_dynamic_inputs()

        return None
