# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_vl_moe.py
#
# Copyright 2025 Huawei Technologites Co., Ltd
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only Qwen3-VL-MoE model compatible with HuggingFace weights."""
import typing
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from typing import Optional

import mindspore as ms
import mindspore.mint as mint
import numpy as np
from mindspore import Parameter, Tensor
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeConfig)
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.models import SupportsPP
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_mindspore.model_executor.layers.logits_processor import (
    LogitsProcessor)
from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead)
from vllm_mindspore.model_executor.model_loader.weight_utils import (
    default_weight_loader, get_loaded_weight)
from vllm_mindspore.model_executor.models.interfaces import (MixtureOfExperts,
                                                             SupportesMoeDpTp)
from vllm_mindspore.model_executor.models.utils import PPMissingLayer

from .qwen2_5_vl import Qwen2_5_VisionRotaryEmbedding
from .qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeModel
from .qwen3_vl import (Qwen3_VisionTransformer, Qwen3VLDummyInputsBuilder,
                       Qwen3VLForConditionalGeneration,
                       Qwen3VLMultiModalProcessor, Qwen3VLProcessingInfo)
from .utils import is_pp_missing_parameter, maybe_prefix

logger = init_logger(__name__)


class Qwen3VLMoeProcessingInfo(Qwen3VLProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3VLMoeConfig)


class Qwen3MoeLLMModel(Qwen3MoeModel):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 deepstack_layers: int = 0):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if not get_pp_group().is_first_rank:
            assert self.start_layer >= len(
                vllm_config.model_config.hf_config.vision_config.
                deepstack_visual_indexes), (
                    "start_layer should be greater than or equal to "
                    "len(deepstack_visual_indexes)")
        self.deepstack_multiscale_layer_start = 1
        self.deepstack_layers = deepstack_layers

    def construct(
        self,
        input_ids: Tensor,
        positions: Tensor,
        key_caches: list[Tensor],
        value_caches: list[Tensor],
        slot_mapping: Tensor,
        attn_mask: Tensor,
        batch_valid_length: Tensor,
        q_seq_lens: Tensor,
        block_tables: Tensor,
        intermediate_hidden_states: Optional[Tensor] = None,
        intermediate_residual: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        dp_pad_index: Optional[Tensor] = None,
        dp_unpad_index: Optional[Tensor] = None,
        dp_pad_index_total_with_offset: Optional[Tensor] = None,
        dp_unpad_index_total_with_offset: Optional[Tensor] = None,
        deepstack_input_embeds: Optional[Mapping[str, Tensor]] = None,
    ) -> tuple[Tensor, Tensor]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            hidden_states = intermediate_hidden_states
            residual = intermediate_residual
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, key_caches[i - self.start_layer],
                value_caches[i - self.start_layer], slot_mapping, attn_mask,
                batch_valid_length, q_seq_lens, block_tables, residual,
                dp_pad_index, dp_unpad_index, dp_pad_index_total_with_offset,
                dp_unpad_index_total_with_offset)

            if deepstack_input_embeds is not None and i in range(
                    self.deepstack_layers):
                hidden_states = mint.add(hidden_states,
                                         deepstack_input_embeds[i])

        if get_pp_group().is_last_rank:
            hidden_states, residual = self.norm(hidden_states, residual)
        return hidden_states, residual

    def load_fused_expert_weights(self, name: str, params_dict: dict,
                                  loaded_weight: Tensor, shard_id: str,
                                  num_experts: int):
        param = params_dict[name]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        for expert_id in range(num_experts):
            curr_expert_weight = loaded_weight[expert_id]
            weight_loader(param, curr_expert_weight, name, shard_id, expert_id)

    def load_weights(self, weights: Iterable[tuple[str, Tensor]],
                     params_dict: dict[str, Parameter]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (".bias", "_bias", ".k_scale", "_k_scale",
                           ".v_scale", "_v_scale", ".weight_scale",
                           "_weight_scale", ".input_scale", "_input_scale")
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]
        num_experts = self.config.num_experts
        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if ("experts.gate_up_proj" in name
                        or "experts.down_proj" in name):
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if is_fused_expert:
                        if name_mapped not in params_dict:
                            continue
                        loaded_weight = get_loaded_weight(loaded_weight)
                        loaded_weight = loaded_weight.swapaxes(-1,
                                                               -2)  # no bias
                        if "experts.gate_up_proj" in name:
                            loaded_weight = np.array_split(loaded_weight,
                                                           2,
                                                           axis=-2)
                            self.load_fused_expert_weights(
                                name_mapped, params_dict, loaded_weight[0],
                                "w1", num_experts)
                            self.load_fused_expert_weights(
                                name_mapped, params_dict, loaded_weight[1],
                                "w3", num_experts)
                        else:
                            self.load_fused_expert_weights(
                                name_mapped, params_dict, loaded_weight,
                                shard_id, num_experts)
                    else:
                        # Skip loading extra parameters for GPTQ/modelopt models
                        if name_mapped.endswith(
                                ignore_suffixes
                        ) and name_mapped not in params_dict:
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # other available replicas.
                        weight_loader = typing.cast(Callable[..., bool],
                                                    param.weight_loader)
                        weight_loader(param,
                                      loaded_weight,
                                      name_mapped,
                                      shard_id=shard_id,
                                      expert_id=expert_id)
                    name = name_mapped
                    break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue
                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(
                            ignore_suffixes) and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale")
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv scale in the checkpoint (e.g. %s), "
                                "but not found the expected name in the model "
                                "(e.g. %s). kv-scale is not loaded.",
                                name,
                                remapped_kv_scale_name,
                            )
                            continue
                        else:
                            name = remapped_kv_scale_name
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3MoeLLMForCausalLM(Qwen3MoeForCausalLM):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 deepstack_layers: int = 0):
        super(Qwen3MoeForCausalLM, self).__init__(vllm_config=vllm_config)
        self.config = vllm_config.model_config.hf_config.text_config
        self.quant_config = vllm_config.quant_config
        self.model = Qwen3MoeLLMModel(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "model"),
                                      deepstack_layers=deepstack_layers)
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(self.config.vocab_size,
                                          self.config.hidden_size,
                                          quant_config=self.quant_config)
        else:
            self.lm_head = PPMissingLayer()
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3VLMoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3VLMoeForConditionalGeneration(Qwen3VLForConditionalGeneration,
                                         SupportesMoeDpTp, MixtureOfExperts,
                                         SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3VLForConditionalGeneration,
              self).__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.multimodal_config = multimodal_config

        self.vision_config = config.vision_config
        self.text_config = config.text_config

        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        if not multimodal_config.get_limit_per_prompt("image") and \
            not multimodal_config.get_limit_per_prompt("video"):
            self.visual = None
        else:
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )
            self.visual.set_model_inputs()
            self.visual.construct = ms.jit(function=self.visual,
                                           jit_level='O0')

        self.use_deepstack = hasattr(config.vision_config,
                                     'deepstack_visual_indexes')
        self.deepstack_num_level = len(
            config.vision_config.deepstack_visual_indexes
        ) if self.use_deepstack else 0

        self.language_model = Qwen3MoeLLMForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            deepstack_layers=self.deepstack_num_level)
        self.model = self.language_model.model
        self.lm_head = self.language_model.lm_head
        self.common_preprocess(vllm_config, prefix)

        self.model.embed_tokens._set_jit_graph_name("prefill")
        self.model.embed_tokens.phase = "prefill"
        dyn_input_ids = ms.Tensor(shape=[None], dtype=ms.int32)
        self.model.embed_tokens.set_inputs(dyn_input_ids)
        self.model.embed_tokens.construct = ms.jit(
            function=self.model.embed_tokens, jit_level='O0')

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

        # register buffer for deepstack
        if self.use_deepstack and self.visual is not None:
            self.deepstack_input_embeds = [
                mint.zeros(
                    (vllm_config.scheduler_config.max_num_batched_tokens,
                     config.text_config.hidden_size),
                    dtype=self.model_config.dtype)
                for _ in range(self.deepstack_num_level)
            ]
        else:
            self.deepstack_input_embeds = None
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level
        head_dim = (self.vision_config.hidden_size //
                    self.vision_config.num_heads)
        self.rotary_pos_emb_full = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        # Use OrderedDict to implement LRUCache for rot_pos_ids
        self._rot_pos_ids_cache: OrderedDict[tuple[int, int],
                                             ms.Tensor] = OrderedDict()
        # Default Cache size set to 256.
        # Assume the Typical image pixel size is 4096 * 4096
        # with patch_size is 14,
        # then pos_ids length will be ((4096 * 4096) / (14 * 14)) * 2 = 85598.
        # Therefore, one cache item will take 85598 * 4(init32) = 342392 Byte
        # With cache max size 256, The cache will maximum occupy 87MB
        self._rot_pos_ids_cache_max_size = int(
            vllm_config.additional_config.get("mm_rot_pos_ids_cache_max_size",
                                              256))
