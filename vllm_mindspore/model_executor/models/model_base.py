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
from typing import Iterable, List, Optional, Set, Tuple, Union, Dict
import numpy as np

import mindspore as ms
from mindspore import Tensor, mutable, nn

from vllm.attention.backends.abstract import AttentionType
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
import vllm.envs as envs

import mindspore as ms
from mindspore import Tensor, nn, mutable
from mindspore.common import dtype as mstype

from vllm_mindspore.model_executor.models.attention_mask import LowerTriangularMask
from vllm_mindspore.utils import STR_DTYPE_TO_MS_DTYPE
from vllm_mindspore.v1.attention.backends.ms_attn import MsAttentionMetadata


class AttentionWrapper:
    def __init__(self):
        vllm_config = get_current_vllm_config()
        block_size = vllm_config.cache_config.block_size
        num_kv_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config)
        head_size = vllm_config.model_config.get_head_size()
        num_block = 0
        self.kv_shape = [num_block, block_size, num_kv_heads, head_size]
        self.kv_cache = [
            (
                ms.mint.zeros(self.kv_shape, dtype=vllm_config.model_config.dtype),
                ms.mint.zeros(self.kv_shape, dtype=vllm_config.model_config.dtype),
            )
            for _ in range(vllm_config.parallel_config.pipeline_parallel_size)
        ]
        self.attn_type = AttentionType.DECODER

        # add for v1
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.dtype = vllm_config.model_config.dtype
        self.block_size = block_size
        self.sliding_window = None


class MLAAttentionWrapper(AttentionWrapper):
    def __init__(self):
        super().__init__()
        vllm_config = get_current_vllm_config()
        self.kv_cache = [
            (ms.mint.zeros(self.kv_shape, dtype=vllm_config.model_config.dtype),)
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
        self.enable_micro_batch = \
                vllm_config.additional_config.get('enable_micro_batch', 0) == 1 \
                if vllm_config.additional_config is not None else False

        self.modules_dict = None

        self.enable_chunked_prefill = vllm_config.scheduler_config.enable_chunked_prefill
        self.enable_prefix_caching = vllm_config.cache_config.enable_prefix_caching
        self.is_multi_step = vllm_config.scheduler_config.is_multi_step
        self.is_multi_step_chunked_prefill = self.is_multi_step and self.enable_chunked_prefill

        self.set_flags = False

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
        raise NotImplementedError("Function load_weights should be Implemented!")


    def _dummy_attention_metadata(self, input_ids: Tensor, positions: Tensor):
        input_len = input_ids.shape[0]
        max_seq_len = ms.Tensor(input_len, dtype=ms.int32)
        seq_lengths = ms.Tensor([input_len], dtype=ms.int32)
        q_seq_lens_np = np.array([input_len], dtype=np.int32)
        seq_lens_np = np.array([input_len], dtype=np.int32)
        context_lens_tensor = ms.Tensor([0], dtype=ms.int32)

        block_tables = ms.Tensor([[0]], dtype=ms.int32)
        slot_mapping = [-1 for _ in range(input_len)]
        slot_mapping = ms.Tensor(slot_mapping, dtype=ms.int32)
        return MsAttentionMetadata(
            max_seq_len=max_seq_len,
            seq_lens=seq_lengths,
            seq_lens_np=seq_lens_np,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            q_seq_lens_np=q_seq_lens_np,
            context_lens=context_lens_tensor,
            # To enforce prefill and decode are both complied in warmup process.
            # So set max_context_lens to 0 for prefill and 1 for decode.
            max_context_lens=0 if not self.set_flags else 1,
            query_start_loc = None
        )


    def prepare_base_inputs(self, input_ids, positions):
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            attn_metadata = self._dummy_attention_metadata(input_ids, positions)
        key_cache, value_cache = self.get_kvcache()
        if not envs.VLLM_USE_V1:
            # V0
            seq_lens = attn_metadata.seq_lens
            max_query_len = attn_metadata.max_query_len
            # When Mutli-Step is enabled with Chunked-Prefill, prefills and
            # decodes are scheduled together. In the first step, all the
            # prefills turn into decodes and max_query_len will be 1.
            if self.is_multi_step_chunked_prefill and max_query_len == 1:
                query_lens = [1] * len(seq_lens)
            else:
                query_lens = attn_metadata.query_lens

            seq_lens_np = np.array(seq_lens, dtype=np.int32)
            query_lens_np = np.array(query_lens, dtype=np.int32)
            kv_cache_lens = seq_lens_np - query_lens_np
            if attn_metadata.num_decode_tokens == 0 and kv_cache_lens.max() == 0:
                is_prefill = True
            else:
                is_prefill = False
        else:
            # V1
            is_prefill = attn_metadata.max_context_lens == 0
            query_lens_np = attn_metadata.q_seq_lens_np
            seq_lens_np = attn_metadata.seq_lens_np
        
        q_seq_lens = ms.Tensor(query_lens_np, dtype=ms.int32)
        position_ids = ms.Tensor(positions, dtype=ms.int32)
        attention_mask = self.casual_mask.gen_attention_mask(is_prefill, positions, query_lens_np)

        model_inputs = {}
        model_inputs["input_ids"] = input_ids.astype(ms.int32)
        model_inputs["batch_valid_length"] = ms.from_numpy(seq_lens_np)
        model_inputs["block_tables"] = attn_metadata.block_tables
        model_inputs["slot_mapping"] = attn_metadata.slot_mapping
        model_inputs["position_ids"] = position_ids
        model_inputs["q_seq_lens"] = q_seq_lens
        model_inputs["attention_mask"] = attention_mask
        model_inputs["key_cache"] = key_cache
        model_inputs["value_cache"] = value_cache

        return model_inputs, is_prefill


class NativeModel(MsModelBase):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.quant_config = vllm_config.quant_config
        if vllm_config.lora_config is not None:
            # native model lora only support pynative mode now
            vllm_config.model_config.enforce_eager = True
        self.is_graph_mode = False if vllm_config.model_config.enforce_eager else True
        self.prev_prefill = False
        self.run_model = None

    def common_preprocess(self, vllm_config, prefix = ""):
        self.set_modules({"model": self.model, "lm_head": self.lm_head})

        self.casual_mask = LowerTriangularMask(dtype=self.model_config.dtype, 
                                               max_model_len=self.model_config.max_model_len)
        self.kv_caches = [AttentionWrapper() for i in range(self.config.num_hidden_layers)]

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(self.config.num_hidden_layers):
            compilation_config.static_forward_context[str(i)] = self.kv_caches[i]


    def set_model_inputs(self, is_prefill):
        dyn_input_ids = Tensor(shape=[None], dtype=mstype.int32)
        dyn_position_ids = Tensor(shape=[None], dtype=mstype.int32)

        block_size = self.cache_config.block_size
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()
        kv_cache_shape = (None, block_size, num_kv_heads, head_size)

        kv_cache_dtype = self.model_config.dtype if self.cache_config.cache_dtype == "auto" \
            else self.cache_config.cache_dtype
        if kv_cache_dtype in STR_DTYPE_TO_MS_DTYPE:
            kv_cache_dtype = STR_DTYPE_TO_MS_DTYPE[kv_cache_dtype]

        num_layers = self.model_config.get_num_layers(self.parallel_config)

        dyn_key_cache = Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype)
        dyn_value_cache = Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype)
        dyn_key_caches = mutable([dyn_key_cache for _ in range(num_layers)])
        dyn_value_caches = mutable([dyn_value_cache for _ in range(num_layers)])

        dyn_slot_mapping = Tensor(shape=[None, ], dtype=mstype.int32)
        dynamic_attention_mask = Tensor(shape=[None, None], dtype=self.model_config.dtype)
        dyn_batch_valid_length = Tensor(shape=[None,], dtype=mstype.int32)
        dyn_q_seq_lens = Tensor(shape=[None, ], dtype=mstype.int32)
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
            dynamic_attention_mask,
            dyn_batch_valid_length,
            dyn_q_seq_lens,
            dyn_block_tables,
            dyn_intermediate_tensors,
            dyn_inputs_embeds
        )

    def prepare_inputs(self, input_ids, positions, intermediate_tensors, inputs_embeds):
        model_inputs, is_prefill = self.prepare_base_inputs(input_ids, positions)

        # for multimodal model
        model_inputs["intermediate_tensors"] = intermediate_tensors
        model_inputs["inputs_embeds"] = inputs_embeds

        return model_inputs, is_prefill

    def exec_model(
        self,
        input_ids: Tensor,
        positions: Tensor,
        intermediate_tensors: IntermediateTensors = None,
        inputs_embeds: Tensor = None,
        **kwargs
    ):
        model_inputs, is_prefill = self.prepare_inputs(input_ids, positions, intermediate_tensors, inputs_embeds)

        if self.prev_prefill != is_prefill and self.is_graph_mode:
            self.set_model_inputs(is_prefill)
        self.prev_prefill = is_prefill

        # for dummy_attention_metadata 
        if is_prefill and not self.set_flags:
            self.set_flags = True

        if self.run_model is None:
            self.run_model = ms.jit(function=self.model, jit_level='O0') if self.is_graph_mode else self.model
        model_output = self.run_model(
            input_ids=model_inputs["input_ids"],
            positions=model_inputs["position_ids"],
            key_caches=model_inputs["key_cache"],
            value_caches=model_inputs["value_cache"],
            is_prefill=is_prefill,
            slot_mapping=model_inputs["slot_mapping"],
            attn_mask=model_inputs["attention_mask"],
            batch_valid_length=model_inputs["batch_valid_length"],
            q_seq_lens=model_inputs["q_seq_lens"],
            block_tables=model_inputs["block_tables"],
            intermediate_tensors=model_inputs["intermediate_tensors"],
            inputs_embeds=model_inputs["inputs_embeds"],
            )

        return model_output
