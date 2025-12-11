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
import torch_npu
import mindspore as ms
import vllm.envs as envs
from mindspore import Tensor, mutable, nn
from mindspore.common import dtype as mstype
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.selector import get_attn_backend
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_dp_group, get_ep_group
from vllm.forward_context import get_forward_context
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.attention.layer import Attention
from vllm.distributed.parallel_state import get_world_group
from vllm.distributed import get_dp_group, get_ep_group

from vllm_mindspore.model_executor.models.attention_mask import (
    LowerTriangularMask)
from vllm_mindspore.model_executor.models.utils import (convert_pin,
                                                        is_use_ringmla)
from vllm_mindspore.model_executor.utils import set_model_context
from vllm_mindspore.utils import (STR_DTYPE_TO_MS_DTYPE, create_kv_cache,
                                  is_310p)
from vllm_mindspore.hybrid_adapter.tensor_convert import (tensor_torch2ms,
    tensor_ms2torch, get_ms_dtype)
from vllm_mindspore.model_executor.models.model_base import NativeModel
from vllm_mindspore.hybrid_adapter.utils import init_ms_distributed
from vllm_mindspore.model_executor.models.interfaces import (
    is_mixture_of_experts, supports_moe_dp_tp)

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
        self.attn_backend = get_attn_backend(head_size,
                                             vllm_config.model_config.dtype,
                                             vllm_config.model_config.dtype,
                                             block_size, False,
                                             vllm_config.model_config.use_mla,
                                             False)

    def get_attn_backend(self):
        return self.attn_backend



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
        if not envs.VLLM_USE_V1:
            raise NotImplementedError("Unsupport V0")

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

        # aclgraph need construct block table with real input
        # here must know max_model_len and block_size
        # for speculative infer, this value need update
        self.block_size = self.cache_config.block_size
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_block_num = int(self.max_model_len / self.block_size)

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

        if is_mixture_of_experts(self):
            self.init_moe_params(vllm_config)

        # add local ms kv-cache
        self.ms_key_caches = []
        self.ms_value_caches = []

        # add for ms communication
        self.rank = torch.distributed.get_rank()
        self.local_rank = get_world_group().local_rank

        init_ms_distributed(self.parallel_config, self.rank, self.local_rank)

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

        # only for V1
        if is_warm_up is True:
            is_prefill = attn_metadata.max_context_lens == 0
            query_lens_np = attn_metadata.q_seq_lens_np
            seq_lens_np = attn_metadata.seq_lens_np
        else:
            is_prefill = attn_metadata.attn_state == 0  # PrefillNoCache
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
            ms.Tensor(seq_lens_np, dtype=ms.int32))
        model_inputs["block_tables"] = convert_pin(tensor_torch2ms(
            attn_metadata.block_tables))
        model_inputs["slot_mapping"] = convert_pin(tensor_torch2ms(
            attn_metadata.slot_mapping).astype(mstype.int32))
        model_inputs["positions"] = convert_pin(position_ids)
        model_inputs["q_seq_lens"] = convert_pin(q_seq_lens)
        model_inputs["attn_mask"] = convert_pin(attention_mask)
        model_inputs["key_caches"] = key_cache
        model_inputs["value_caches"] = value_cache
        # for multimodal model
        if intermediate_tensors is not None:
            model_inputs["input_ids"] = None
            model_inputs[
                "intermediate_hidden_states"] = intermediate_tensors[
                    "hidden_states"]
            model_inputs["intermediate_residual"] = intermediate_tensors[
                "residual"]
        else:
            model_inputs["intermediate_hidden_states"] = None
            model_inputs["intermediate_residual"] = None
        model_inputs["inputs_embeds"] = inputs_embeds

        if supports_moe_dp_tp(self):
            dp_unpad_index, dp_pad_index, dp_pad_index_total_with_offset, \
            dp_unpad_index_total_with_offset = self.prepare_moe_dp_tp_inputs()
            model_inputs["dp_unpad_index"] = dp_unpad_index
            model_inputs["dp_pad_index"] = dp_pad_index
            model_inputs["dp_pad_index_total_with_offset"] = \
                dp_pad_index_total_with_offset
            model_inputs["dp_unpad_index_total_with_offset"] = \
                dp_unpad_index_total_with_offset
        return model_inputs, is_prefill

    def common_preprocess(self, vllm_config, prefix=""):
        self.set_modules({"model": self.model, "lm_head": self.lm_head})

        self.casual_mask = LowerTriangularMask(
            dtype=self.model_config.dtype,
            max_model_len=self.model_config.max_model_len)
        self.kv_caches = [
            # difference AttentionWrapper
            AttentionWrapper() for i in range(self.num_layers)
        ]

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(self.num_layers):
            compilation_config.static_forward_context[str(
                i)] = self.kv_caches[i]

    def exec_model(self,
                   input_ids,
                   positions,
                   intermediate_tensors=None,
                   inputs_embeds=None,
                   **kwargs):
        outputs = super().exec_model(input_ids, positions,
                                     intermediate_tensors,
                                     inputs_embeds, **kwargs)
        new_outputs = []
        for output in outputs:
            new_outputs.append(tensor_ms2torch(output))
        return new_outputs

    def compute_logits(
        self,
        hidden_states,
    ):
        ms_hidden_states = tensor_torch2ms(hidden_states)
        ms_logits = self._compute_logits(ms_hidden_states)
        logits = tensor_ms2torch(ms_logits)
        return logits
