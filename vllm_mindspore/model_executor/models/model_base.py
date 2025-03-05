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

import os
from abc import abstractmethod
from typing import Iterable, List, Optional, Set, Tuple, Union, Dict

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from mindspore import Tensor, nn, mutable
from mindspore import dtype as mstype

from vllm_mindspore.utils import STR_DTYPE_TO_MS_DTYPE


class MsModelBase():
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super(MsModelBase, self).__init__()
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.model_config = vllm_config.model_config
        self.lora_config = lora_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.load_config = vllm_config.load_config

        self.modules_dict = None

    def get_model_path(self):
        model_name_or_path = self.model_config.model
        if os.path.isdir(model_name_or_path):
            return model_name_or_path
        else:
            from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf
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

        res_modules = set()
        for name, module in self.modules_dict.items():
            for module_name, sub_module in module.cells_and_names():
                if name != "self":
                    module_name = name + "." + module_name
                yield module_name, sub_module

    def get_submodule(self):
        raise RuntimeError("Cannot get submodule for mindspore model now!")

    def eval(self):
        self._check_modules_valid()

        for _, module in self.modules_dict.items():
            module.set_train(False)

        return self

    def __call__(
        self,
        input_ids: Tensor,
        positions: Tensor,
        kv_caches: List[Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[Tensor] = None,
    ) -> Union[Tensor, IntermediateTensors]:
        return self.forward(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors,
            inputs_embeds,
        )

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        kv_caches: List[Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[Tensor] = None,
    ) -> Union[Tensor, IntermediateTensors]:
        raise NotImplementedError

    def set_model_inputs(self):
        dyn_input_ids = Tensor(shape=[None, None], dtype=mstype.int64)
        dyn_position_ids = Tensor(shape=[None], dtype=mstype.int64)

        block_size = self.cache_config.block_size
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()
        kv_cache_shape = (None, block_size, num_kv_heads, head_size)

        kv_cache_dtype = self.model_config.dtype if self.cache_config.cache_dtype == "auto" \
            else self.cache_config.cache_dtype
        kv_cache_dtype = STR_DTYPE_TO_MS_DTYPE[kv_cache_dtype]

        num_layers = self.model_config.get_num_layers(self.parallel_config)

        dyn_key_cache = mutable(Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype))
        dyn_value_cache = mutable(Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype))
        dyn_kv_cache = mutable((dyn_key_cache, dyn_value_cache))
        dyn_kv_caches = mutable([dyn_kv_cache for _ in range(num_layers)])

        dyn_num_prefill_tokens = mutable(1)
        dyn_num_decode_tokens = mutable(0)
        dyn_context_lens = Tensor(shape=[None, ], dtype=mstype.int32)
        dyn_batch_valid_length = mutable([0, 0, 0], dynamic_len=True)
        dyn_slot_mapping = Tensor(shape=[None, ], dtype=mstype.int32)
        dyn_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dyn_intermediate_tensors = None
        dyn_inputs_embeds = None

        self.model.set_inputs(
            dyn_input_ids,
            dyn_position_ids,
            dyn_kv_caches,
            dyn_num_prefill_tokens,
            dyn_num_decode_tokens,
            dyn_context_lens,
            dyn_batch_valid_length,
            dyn_slot_mapping,
            dyn_block_tables,
            dyn_intermediate_tensors,
            dyn_inputs_embeds
        )

    @abstractmethod
    def compute_logits(
        self,
        hidden_states: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[Tensor]:
        raise NotImplementedError("Function compute_logits should be Implemented!")

    @abstractmethod
    def sample(
        self,
        logits: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        raise NotImplementedError("Function sample should be Implemented!")

    @abstractmethod
    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> Set[str]:
        raise NotImplementedError("Function load_weights should be Implemented!")
