#!/usr/bin/env python3
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

import os
from abc import abstractmethod
from typing import Dict, Iterable, Optional, Set, Tuple, Union

import torch
from mindspore import Tensor, mutable, nn
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors


class Fake_Attention:

    def __init__(self):
        vllm_config = get_current_vllm_config()
        block_size = vllm_config.cache_config.block_size
        num_kv_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config)
        head_size = vllm_config.model_config.get_head_size()
        num_block = 0
        self.kv_shape = [num_block, block_size, num_kv_heads, head_size]
        self.kv_cache = [(
            torch.zeros(self.kv_shape, dtype=torch.bfloat16, device="Ascend"),
            torch.zeros(self.kv_shape, dtype=torch.bfloat16, device="Ascend"),
        ) for _ in range(vllm_config.parallel_config.pipeline_parallel_size)]
        self.attn_type = AttentionType.DECODER


class Fake_MLA(Fake_Attention):

    def __init__(self):
        super().__init__()
        vllm_config = get_current_vllm_config()
        self.kv_cache = [
            (torch.zeros(self.kv_shape, dtype=torch.bfloat16,
                         device="Ascend"), )
            for _ in range(vllm_config.parallel_config.pipeline_parallel_size)
        ]


class Fake_Attention_V1(Attention):

    def __init__(self):
        vllm_config = get_current_vllm_config()
        block_size = vllm_config.cache_config.block_size
        num_kv_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config)
        head_size = vllm_config.model_config.get_head_size()
        num_block = 0
        self.kv_shape = [num_block, block_size, num_kv_heads, head_size]
        self.kv_cache = [(
            torch.zeros(self.kv_shape, dtype=torch.bfloat16, device="Ascend"),
            torch.zeros(self.kv_shape, dtype=torch.bfloat16, device="Ascend"),
        ) for _ in range(vllm_config.parallel_config.pipeline_parallel_size)]
        self.attn_type = AttentionType.DECODER
        self.num_block = num_block
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.dtype = vllm_config.model_config.dtype
        self.block_size = block_size
        self.sliding_window = None


class Fake_MLA_V1(Fake_Attention_V1):

    def __init__(self):
        super().__init__()
        vllm_config = get_current_vllm_config()
        self.kv_cache = [
            (torch.zeros(self.kv_shape, dtype=torch.bfloat16,
                         device="Ascend"), )
            for _ in range(vllm_config.parallel_config.pipeline_parallel_size)
        ]


class MsModelBase:

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.model_config = vllm_config.model_config
        self.lora_config = lora_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.load_config = vllm_config.load_config
        self.scheduler_config = vllm_config.scheduler_config

        self.modules_dict = None

        self.enable_chunked_prefill = vllm_config.scheduler_config.enable_chunked_prefill
        self.enable_prefix_caching = vllm_config.cache_config.enable_prefix_caching
        self.is_multi_step = vllm_config.scheduler_config.is_multi_step
        self.is_multi_step_chunked_prefill = self.is_multi_step and self.enable_chunked_prefill

    def get_model_path(self):
        model_name_or_path = self.model_config.model
        if os.path.isdir(model_name_or_path):
            return model_name_or_path
        else:
            from vllm.model_executor.model_loader.weight_utils import \
                download_weights_from_hf
            allow_patterns = ["*.safetensors"]
            revision = self.model_config.revision
            return download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )

    def set_modules(self, model_dicts: Dict[str, nn.Cell]):
        self.modules_dict = model_dicts

    def _check_modules_valid(self):
        if self.modules_dict is None:
            raise RuntimeError("Should set modules firstly!")

    def named_parameters(self):
        self._check_modules_valid()

        for cell_name, module in self.modules_dict.items():
            for par_name, par in module.parameters_and_names():
                if cell_name != "self":
                    par_name = cell_name + "." + par_name

                yield par_name, par

    def get_params_dict(self):
        self._check_modules_valid()

        params_dict = dict()
        for name, module in self.modules_dict.items():
            module_params = module.parameters_dict()
            if name != "self":
                new_module_params = dict()
                for param_name, param in module_params.items():
                    new_module_params[name + "." + param_name] = param
                module_params = new_module_params
            params_dict.update(module_params)

        return params_dict

    def named_modules(self, remove_duplicate: bool = True):
        self._check_modules_valid()

        for name, module in self.modules_dict.items():
            for module_name, sub_module in module.cells_and_names():
                if name != "self":
                    module_name = name + "." + module_name
                yield module_name, sub_module

    def get_submodule(self, target: str):
        parts = target.split(".")
        if target == "":
            return self
        for part in parts:
            if not part:
                raise ValueError(
                    f"Invalid submodule path: empty part in '{target}'")
        current = self
        for part in parts:
            current = getattr(current, part)
        return current

    def eval(self):
        self._check_modules_valid()

        for _, module in self.modules_dict.items():
            module.set_train(False)

        return self

    def __call__(
        self,
        input_ids: Tensor,
        positions: Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[Tensor] = None,
        previous_hidden_states: Optional[Tensor] = None,
        spec_step_idx: int = 0,
    ) -> Union[Tensor, IntermediateTensors]:
        return self.forward(input_ids,
                            positions,
                            intermediate_tensors,
                            inputs_embeds,
                            previous_hidden_states=previous_hidden_states,
                            spec_step_idx=spec_step_idx)

    def forward(self,
                input_ids: Tensor,
                positions: Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[Tensor] = None,
                **kwargs) -> Union[Tensor, IntermediateTensors]:
        raise NotImplementedError

    def set_model_inputs(self, is_prefill):
        dyn_input_ids = Tensor(shape=[None, None], dtype=mstype.int64)
        dyn_position_ids = Tensor(shape=[None], dtype=mstype.int64)

        block_size = self.cache_config.block_size
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()
        kv_cache_shape = (None, block_size, num_kv_heads, head_size)

        kv_cache_dtype = self.model_config.dtype if self.cache_config.cache_dtype == "auto" \
            else self.cache_config.cache_dtype
        if kv_cache_dtype in STR_DTYPE_TO_MS_DTYPE:
            kv_cache_dtype = STR_DTYPE_TO_MS_DTYPE[kv_cache_dtype]

        num_layers = self.model_config.get_num_layers(self.parallel_config)

        dyn_key_cache = mutable(Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype))
        dyn_value_cache = mutable(Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype))
        dyn_key_caches = mutable([dyn_key_cache for _ in range(num_layers)])
        dyn_value_caches = mutable([dyn_value_cache for _ in range(num_layers)])

        dyn_batch_valid_length = Tensor(shape=[None, ], dtype=mstype.int32)
        dyn_q_seq_lens = Tensor(shape=[None, ], dtype=mstype.int32)
        dyn_slot_mapping = Tensor(shape=[None, ], dtype=mstype.int32)
        dyn_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dyn_intermediate_tensors = None
        dyn_inputs_embeds = None

        self.model.set_inputs(
            dyn_input_ids,
            dyn_position_ids,
            dyn_key_caches,
            dyn_value_caches,
            is_prefill,
            dyn_slot_mapping,
            dyn_batch_valid_length,
            dyn_q_seq_lens,
            dyn_block_tables,
            dyn_intermediate_tensors,
            dyn_inputs_embeds
        )

    def get_kvcache(self):
        key_cache = []
        value_cache = []
        forward_context = get_forward_context()
        for i in range(self.config.num_hidden_layers):
            k_cache = self.kv_caches[i].kv_cache[
                forward_context.virtual_engine][0]
            v_cache = self.kv_caches[i].kv_cache[
                forward_context.virtual_engine][1]
            key_cache.append(k_cache)
            value_cache.append(v_cache)
        return mutable(key_cache), mutable(value_cache)

    @abstractmethod
    def compute_logits(
        self,
        hidden_states: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[Tensor]:
        raise NotImplementedError(
            "Function compute_logits should be Implemented!")

    @abstractmethod
    def sample(
        self,
        logits: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        raise NotImplementedError("Function sample should be Implemented!")

    @abstractmethod
    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> Set[str]:
        raise NotImplementedError(
            "Function load_weights should be Implemented!")
