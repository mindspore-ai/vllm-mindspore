# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologies Co., Ltd.
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

from typing import Any, Optional, Union
import copy

import torch
import mindspore as ms
import vllm.envs as envs
from mindspore import Tensor, mutable, nn
from mindspore.common import dtype as mstype
from vllm.attention.backends.abstract import AttentionType
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.sequence import IntermediateTensors
from vllm.attention.layer import Attention

from omni.layers.attention.backend.attention import AscendAttentionState

from vllm_mindspore.model_executor.models.attention_mask import (
    LowerTriangularMask)
from vllm_mindspore.model_executor.models.utils import (convert_pin,
                                                        is_use_ringmla)
from vllm_mindspore.model_executor.utils import set_model_context
from vllm_mindspore.utils import STR_DTYPE_TO_MS_DTYPE, create_kv_cache
from vllm_mindspore.external.tensor_convert import (tensor_torch2ms,
    get_ms_dtype)
from vllm_mindspore.model_executor.models.model_base import NativeModel


class AttentionWrapper(Attention):
    def __init__(self):
        vllm_config = get_current_vllm_config()
        block_size = vllm_config.cache_config.block_size
        num_kv_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config)
        head_size = vllm_config.model_config.get_head_size()
        num_block = 0
        self.ms_dtype = get_ms_dtype(vllm_config.model_config.dtype)
        self.kv_shape = [num_block, block_size, num_kv_heads, head_size]
        self.kv_cache = [
            (create_kv_cache(self.kv_shape, self.ms_dtype),
             create_kv_cache(self.kv_shape, self.ms_dtype))
            for _ in range(vllm_config.parallel_config.pipeline_parallel_size)
        ]

        self.attn_type = AttentionType.DECODER

        # add for v1
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.dtype = self.ms_dtype
        self.block_size = block_size
        self.sliding_window = None
        self.kv_sharing_target_layer_name = None


class MLAAttentionWrapper(AttentionWrapper):
    def __init__(self, fa3_quant=False, kv_cache_dtype=None):
        super().__init__()
        vllm_config = get_current_vllm_config()
        self.use_ringmla = is_use_ringmla(vllm_config)
        if kv_cache_dtype is None:
            kv_cache_dtype = vllm_config.model_config.dtype
        self.dtype = get_ms_dtype(kv_cache_dtype)
        if not self.use_ringmla:
            self.kv_cache = [
                (
                    ms.mint.zeros(
                        self.kv_shape,  # type: ignore[misc]
                        dtype=self.dtype), ) for _ in
                range(vllm_config.parallel_config.pipeline_parallel_size)
            ]
        else:
            kv_lora_rank = getattr(vllm_config.model_config.hf_text_config,
                                   'kv_lora_rank', 0)
            qk_rope_head_dim = getattr(vllm_config.model_config.hf_text_config,
                                       'qk_rope_head_dim', 0)
            # k_shape, r_shape used for mla_op
            if fa3_quant:
                k_shape = [*(self.kv_shape[0:-2]), kv_lora_rank]
                r_shape = [*(self.kv_shape[0:-2]), qk_rope_head_dim]
                self.kv_cache = [(
                    ms.mint.zeros(k_shape, dtype=self.dtype),
                    ms.mint.zeros(r_shape, dtype=self.dtype),
                ) for _ in range(
                    vllm_config.parallel_config.pipeline_parallel_size)]
            else:
                k_shape = [*(self.kv_shape[0:-1]), kv_lora_rank]
                r_shape = [*(self.kv_shape[0:-1]), qk_rope_head_dim]
                self.kv_cache = [
                    (ms.mint.zeros(k_shape,
                                   dtype=self.dtype),
                     ms.mint.zeros(r_shape,
                                   dtype=self.dtype))
                    for _ in range(
                        vllm_config.parallel_config.pipeline_parallel_size)
                ]


class MsModelAdapter(NativeModel):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config
        self.config = config

        # convert to mindspore tensor dtype
        self.model_config = copy.deepcopy(vllm_config.model_config)
        self.model_config.dtype = get_ms_dtype(vllm_config.model_config.dtype)
        set_model_context("model_dtype", self.model_config.dtype)

        self.lora_config = lora_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.load_config = vllm_config.load_config
        self.scheduler_config = vllm_config.scheduler_config

        self.modules_dict: Optional[dict[str, nn.Cell]] = None
        self.num_layers = self.model_config.get_num_layers(
            self.parallel_config)

        self.use_ringmla: bool = False
        self.has_prefill_warmup: bool = False
        self.has_chunked_warmup: bool = not self.use_ringmla
        self.kv_caches: list[Any] = []
        self.model: Optional[nn.Cell] = None
        self.lm_head: Optional[nn.Cell] = None

        self.quant_config = vllm_config.quant_config
        if vllm_config.lora_config is not None:
            # native model lora only support pynative mode now
            vllm_config.model_config.enforce_eager = True
        self.is_eager_mode = vllm_config.model_config.enforce_eager
        self.prefill_graph = None
        self.decode_graph = None

        # add local ms kv-cache
        self.ms_key_caches = []
        self.ms_value_caches = []

    def __call__(
        self,
        input_ids: Tensor,
        positions: Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[Tensor] = None,
        previous_hidden_states: Optional[Tensor] = None,
        spec_step_idx: int = 0,
        **kwargs,
    ) -> Union[Tensor, IntermediateTensors]:
        # convert to ms tensor
        input_ids = tensor_torch2ms(input_ids)
        positions = tensor_torch2ms(positions)
        inputs_embeds = tensor_torch2ms(inputs_embeds)
        previous_hidden_states = tensor_torch2ms(previous_hidden_states)

        return self.forward(input_ids,
                            positions,
                            intermediate_tensors,
                            inputs_embeds,
                            previous_hidden_states=previous_hidden_states,
                            spec_step_idx=spec_step_idx,
                            **kwargs)

    def get_kvcache(self):
        if self.ms_key_caches and self.ms_value_caches:
            return mutable(self.ms_key_caches), mutable(self.ms_value_caches)

        key_cache = []
        value_cache = []
        forward_context = get_forward_context()
        for i in range(self.num_layers):
            k_cache = self.kv_caches[i].kv_cache[
                forward_context.virtual_engine][0]
            v_cache = self.kv_caches[i].kv_cache[
                forward_context.virtual_engine][1]
            key_cache.append(tensor_torch2ms(k_cache))
            value_cache.append(tensor_torch2ms(v_cache))

        if not self.has_prefill_warmup:
            return mutable(key_cache), mutable(value_cache)
        else:
            self.ms_key_caches = key_cache
            self.ms_value_caches = value_cache
            return mutable(self.ms_key_caches), mutable(self.ms_value_caches)


    def prepare_inputs(self, input_ids, positions, intermediate_tensors,
                       inputs_embeds):
        is_warm_up = False
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            is_warm_up = True
            attn_metadata = self._dummy_attention_metadata(
                input_ids, positions)
        elif isinstance(attn_metadata, dict) and '1' in attn_metadata:
            attn_metadata = attn_metadata['1']
        key_cache, value_cache = self.get_kvcache()
        if not envs.VLLM_USE_V1:
            raise NotImplementedError("Unsupport V0")
        else:
            # V1
            if is_warm_up is True:
                is_prefill = attn_metadata.max_context_lens == 0
                query_lens_np = attn_metadata.q_seq_lens_np
                seq_lens_np = attn_metadata.seq_lens_np
            else:
                if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
                    is_prefill = True
                else:
                    is_prefill = False
                query_lens_np = attn_metadata.query_lens.cpu().numpy()
                seq_lens_np = attn_metadata.seq_lens.cpu().numpy()

        if input_ids is not None:
            input_ids = input_ids.astype(ms.int32)
        q_seq_lens = ms.Tensor(query_lens_np, dtype=ms.int32)
        position_ids = ms.Tensor(positions, dtype=ms.int32)
        attention_mask = self.casual_mask.gen_attention_mask(
            is_prefill, position_ids, query_lens_np, seq_lens_np)

        model_inputs = {}
        model_inputs["input_ids"] = convert_pin(input_ids)
        model_inputs["batch_valid_length"] = convert_pin(
            ms.from_numpy(seq_lens_np).astype(mstype.int32))
        model_inputs["block_tables"] = tensor_torch2ms(
            attn_metadata.block_tables)
        model_inputs["slot_mapping"] = tensor_torch2ms(
            attn_metadata.slot_mapping).astype(mstype.int32)
        model_inputs["positions"] = convert_pin(position_ids)
        model_inputs["q_seq_lens"] = convert_pin(q_seq_lens)
        model_inputs["attn_mask"] = convert_pin(attention_mask)
        model_inputs["key_caches"] = key_cache
        model_inputs["value_caches"] = value_cache
        # for multimodal model
        model_inputs["intermediate_tensors"] = intermediate_tensors
        model_inputs["inputs_embeds"] = inputs_embeds

        return model_inputs, is_prefill

    def common_preprocess(self, vllm_config, prefix=""):
        self.set_modules({"model": self.model, "lm_head": self.lm_head})

        self.casual_mask = LowerTriangularMask(
            dtype=self.model_config.dtype,
            max_model_len=self.model_config.max_model_len)
        self.kv_caches = [
            # difference AttentionWrapper
            AttentionWrapper() for i in range(self.config.num_hidden_layers)
        ]

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(self.config.num_hidden_layers):
            compilation_config.static_forward_context[str(
                i)] = self.kv_caches[i]
    
    def convert_logits(self, logits):

        # logits = tensor_ms2torch(logits).npu()
        logits = torch.tensor(logits.astype(mstype.float32).asnumpy(),
                              dtype=torch.bfloat16).npu()
        return logits
