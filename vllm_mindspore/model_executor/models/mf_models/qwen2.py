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
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP

from mindspore import Tensor, mutable
from mindspore.nn.utils import no_init_parameters

from mindformers.models.llama import LlamaConfig as LlamaConfig_MF
from research.qwen2_5.infer.qwen2_5 import (
    ParallelQwenForCausalLM as ParallelQwenForCausalLM_MF,
)

from vllm_mindspore.model_executor.layers.sampler import get_sampler
from vllm_mindspore.model_executor.models.model_base import Fake_Attention
from vllm_mindspore.model_executor.models.mf_models.mf_model_base import MfModelBase
from vllm_mindspore.model_executor.models.mf_models.qwen2_weight_processor import Qwen2WeightProcessor
from vllm_mindspore.model_executor.models.mf_models.attention_mask import LowerTriangularMask
from vllm_mindspore.model_executor.models.utils import make_empty_intermediate_tensors_factory
import os
import mindspore as ms


logger = init_logger(__name__)

def set_runtime_kernel_launch_group():
    kernel_launch_group = {'thread_num' : 2, 'kernel_group_num' : 8}
    env_kernel_launch_group = os.getenv("EXPERIMENTAL_KERNEL_LAUNCH_GROUP", None)
    if env_kernel_launch_group == None:
        return
    if env_kernel_launch_group is not None:
        pairs = env_kernel_launch_group.split(',')
        for pair in pairs:
            key, val = pair.split(':')
            kernel_launch_group[key] = val
    thread_num = int(kernel_launch_group.get('thread_num', 2))
    kernel_group_num = int(kernel_launch_group.get('kernel_group_num', 8))
    ms.runtime.set_kernel_launch_group(thread_num=thread_num, kernel_group_num=kernel_group_num)

class Qwen2ForCausalLM(MfModelBase, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super(Qwen2ForCausalLM, self).__init__(vllm_config=vllm_config, prefix=prefix)
        self.mf_kvcaches_init = False

        self.sampler = get_sampler()
        self.set_modules({"model": self.network})
        self.num_layers = self.model_config.get_num_layers(self.parallel_config)

        self.kv_caches = [Fake_Attention() for _ in range(self.num_layers)]
        compilation_config = get_current_vllm_config().compilation_config

        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(self.num_layers):
            compilation_config.static_forward_context[str(i)] = self.kv_caches[i]

        set_runtime_kernel_launch_group()
        self.casual_mask = LowerTriangularMask(mf_model_config=self.mf_model_config)
        self.set_flags = False
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ['hidden_states'], self.mf_model_config.hidden_size
        )

    def get_kvcache(self):
        key_cache = []
        value_cache = []
        forward_context = get_forward_context()
        for i in range(self.num_layers):
            k_cache = self.kv_caches[i].kv_cache[forward_context.virtual_engine][0]
            v_cache = self.kv_caches[i].kv_cache[forward_context.virtual_engine][1]
            key_cache.append(k_cache)
            value_cache.append(v_cache)
        return mutable(key_cache), mutable(value_cache)

    def _generate_model_config(self):
        self.mf_config.load_checkpoint = self.get_model_path()
        self.mf_model_config = LlamaConfig_MF(**self.mf_config.model.model_config)
        if self.mf_config.moe_config:
            self.mf_model_config.moe_config = self.mf_config.moe_config
        self.mf_model_config.return_hidden_states = True

        # qwen qkv concat will support in next version
        # self.mf_model_config.qkv_concat = True
        setattr(self.mf_model_config, 'npu_mem_size', -1)
        # self.mf_config.model.model_config.qkv_concat = True

    def _create_network(self):
        # Initial network
        with no_init_parameters():  # Delay initialization
            network = ParallelQwenForCausalLM_MF(self.mf_model_config)

        if get_pp_group().is_last_rank:
            return network, network.lm_head
        return network, None

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> Set[str]:
        weight_processor = Qwen2WeightProcessor(self.mf_config, self.network, False)
        weight_processor.load_safetensors_shard(self.mf_config.load_checkpoint)

        # self.network.set_dynamic_inputs()
        # dynamic_hidden_states = Tensor(shape=[None, None], dtype=self.mf_model_config.compute_dtype)
        # self.lm_head.set_inputs(dynamic_hidden_states)
        return None
