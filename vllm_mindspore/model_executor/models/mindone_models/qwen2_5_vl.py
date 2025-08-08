# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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
# isort: skip_file
"""Inference-only Qwen2.5-VL model compatible with HuggingFace weights."""

from collections.abc import Iterable, Mapping

# yapf:disable
from functools import cached_property, partial
from typing import Any, Optional, Union

import mindspore
import numpy as np
from mindone.transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel)
from mindone.transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration as MindONE_Qwen2_5_VLForConditionalGeneration)  # noqa: E501
from mindone.transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLPreTrainedModel)
from mindspore import Tensor, mint, mutable, nn, ops
from mindspore.common.api import _pynative_executor
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLConfig, Qwen2_5_VLDummyInputsBuilder)
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration as vLLM_Qwen2_5_VLForConditionalGeneration)  # noqa: E501
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLImageEmbeddingInputs, Qwen2_5_VLImageInputs,
    Qwen2_5_VLImagePixelInputs)
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLMultiModalProcessor as vLLM_Qwen2_5_VLMultiModalProcessor)
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLProcessingInfo, Qwen2_5_VLVideoEmbeddingInputs,
    Qwen2_5_VLVideoInputs, Qwen2_5_VLVideoPixelInputs)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import PromptReplacement
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import uses_mrope

from vllm_mindspore.model_executor.models.attention_mask import (
    LowerTriangularMask)
from vllm_mindspore.model_executor.models.interfaces import SupportsMultiModal
from vllm_mindspore.model_executor.models.mindone_models.qwen2 import (
    MindONEModelBase)
from vllm_mindspore.model_executor.models.mindone_models.qwen2 import (
    Qwen2ForCausalLM as vLLM_Qwen2ForCausalLM)
from vllm_mindspore.model_executor.models.mindone_models.qwen2 import (
    vLLMQwen2Model)
from vllm_mindspore.model_executor.models.mindone_models.utils import (
    enable_dynamic_shape)
from vllm_mindspore.model_executor.models.model_base import Fake_Attention
from vllm_mindspore.model_executor.models.utils import (
    maybe_prefix, merge_multimodal_embeddings)
from vllm_mindspore.model_executor.sampling_metadata import SamplingMetadata
from vllm_mindspore.utils import STR_DTYPE_TO_MS_DTYPE

# yapf:enable


class Qwen2ForCausalLM(vLLM_Qwen2ForCausalLM):
    # rewrite __init__
    def __init__(self,
                 mindone_model: nn.Cell,
                 vllm_config: VllmConfig,
                 prefix: str = ""):
        MindONEModelBase.__init__(self, vllm_config=vllm_config, prefix=prefix)

        # create model
        self.model, self.lm_head = mindone_model.model, mindone_model.lm_head

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        self.sampler = get_sampler()

        self.set_modules({"model": self.model, "lm_head": self.lm_head})
        self.kv_caches = [
            Fake_Attention() for i in range(config.num_hidden_layers)
        ]
        compilation_config = vllm_config.compilation_config

        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(config.num_hidden_layers):
            compilation_config.static_forward_context[str(
                i)] = self.kv_caches[i]


class Qwen2_5_VLMultiModalProcessor(vLLM_Qwen2_5_VLMultiModalProcessor):
    # === Multi-Model Processor === #
    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(
            **hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        placeholder = {
            "image": vocab[hf_processor.image_token],
            "video": vocab[hf_processor.video_token],
        }

        merge_length = image_processor.merge_size**2

        def get_replacement_qwen2vl(item_idx: int, modality: str):
            grid_thw = out_mm_kwargs[f"{modality}_grid_thw"][item_idx]
            assert isinstance(grid_thw, (mindspore.Tensor, np.ndarray))

            if isinstance(grid_thw, np.ndarray):
                num_tokens = int(np.prod(grid_thw)) // merge_length
            else:
                num_tokens = int(grid_thw.prod()) // merge_length

            return [placeholder[modality]] * num_tokens

        return [
            PromptReplacement(
                modality=modality,
                target=[placeholder[modality]],
                replacement=partial(get_replacement_qwen2vl,
                                    modality=modality),
            ) for modality in ("image", "video")
        ]


class NEW_MindONE_Qwen2_5_VLForConditionalGeneration(
        MindONE_Qwen2_5_VLForConditionalGeneration):
    # === Multi-Model Model === #
    # replace qwen2 model to vLLM (with PA)
    def __init__(self, config):
        Qwen2_5_VLPreTrainedModel.__init__(self, config)

        self.visual = Qwen2_5_VisionTransformerPretrainedModel(
            config.vision_config)
        self.model = vLLMQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = mint.nn.Linear(config.hidden_size,
                                      config.vocab_size,
                                      bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder,
)
class Qwen2_5_VLForConditionalGeneration(MindONEModelBase, SupportsMultiModal):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config: Qwen2_5_VLConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        mindone_model = NEW_MindONE_Qwen2_5_VLForConditionalGeneration.from_pretrained(  # noqa: E501
            vllm_config.model_config.model,
            mindspore_dtype=mindspore.bfloat16)

        self.visual = mindone_model.visual
        self.language_model = Qwen2ForCausalLM(
            mindone_model, vllm_config, maybe_prefix(prefix, "language_model"))

        self.set_modules({
            "visual": self.visual,
            "language_model": self.language_model
        })
        self.prefill = True
        self.mstype = STR_DTYPE_TO_MS_DTYPE.get(self.model_config.dtype,
                                                self.model_config.dtype)
        self.casual_mask = LowerTriangularMask(
            dtype=self.mstype, max_model_len=self.model_config.max_model_len)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                        name: str) -> mindspore.Tensor:
        if not isinstance(mm_input, (mindspore.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                             f"Got type: {type(mm_input)}")
        if isinstance(mm_input, mindspore.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(f"{name} should be 2D or batched 3D tensor. "
                                 f"Got ndim: {mm_input.ndim} "
                                 f"(shape={mm_input.shape})")
            return ops.concat(list(mm_input))
        else:
            return ops.concat(mm_input)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Qwen2_5_VLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(pixel_values, (mindspore.Tensor, list)):
                raise ValueError("Incorrect type of image pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return Qwen2_5_VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            image_embeds = self._validate_and_reshape_mm_tensor(
                image_embeds, "image embeds")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(image_embeds, mindspore.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            return Qwen2_5_VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

        return None

    def _parse_and_validate_video_input(
            self, **kwargs: object) -> Optional[Qwen2_5_VLVideoInputs]:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            pixel_values_videos = self._validate_and_reshape_mm_tensor(
                pixel_values_videos, "video pixel values")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            return Qwen2_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )

        if video_embeds is not None:
            video_embeds = self._validate_and_reshape_mm_tensor(
                video_embeds, "video embeds")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            if not isinstance(video_embeds, mindspore.Tensor):
                raise ValueError("Incorrect type of video embeddings. "
                                 f"Got type: {type(video_embeds)}")
            return Qwen2_5_VLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw,
            )

        return None

    _process_image_input = vLLM_Qwen2_5_VLForConditionalGeneration._process_image_input  # noqa: E501
    _process_video_input = vLLM_Qwen2_5_VLForConditionalGeneration._process_video_input  # noqa: E501
    _parse_and_validate_multimodal_inputs = (
        vLLM_Qwen2_5_VLForConditionalGeneration.
        _parse_and_validate_multimodal_inputs)
    get_multimodal_embeddings = None

    def get_input_embeddings(
        self,
        input_ids: mindspore.Tensor,
        multimodal_embeddings: Optional[tuple[mindspore.Tensor, ...]] = None,
    ) -> mindspore.Tensor:
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                [self.config.image_token_id, self.config.video_token_id],
            )
        return inputs_embeds

    def get_input_embeddings_v0(
        self,
        input_ids: mindspore.Tensor,
        image_input: Optional[tuple[mindspore.Tensor, ...]] = None,
        video_input: Optional[tuple[mindspore.Tensor, ...]] = None,
    ) -> mindspore.Tensor:
        inputs_embeds = self.get_input_embeddings(input_ids)
        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds,
                placeholder_token_id=self.config.image_token_id,
            )

        if video_input is not None:
            video_embeds = self._process_video_input(video_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                video_embeds,
                placeholder_token_id=self.config.video_token_id,
            )
        return inputs_embeds

    def run_language_model(
        self,
        input_ids: Tensor,
        positions: Tensor,
        kv_caches: list[tuple[Tensor, Tensor]],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: IntermediateTensors = None,
        inputs_embeds: Tensor = None,
    ):
        key_caches, value_caches = self.language_model.get_kvcache()

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
        is_prefill = bool(attn_metadata.num_decode_tokens == 0
                          and kv_cache_lens.max() == 0)

        if is_prefill > 0:
            if input_ids is not None:
                input_ids = input_ids.expand_dims(0)
            else:
                inputs_embeds = inputs_embeds.expand_dims(0)
        else:
            if input_ids is not None:
                input_ids = input_ids.expand_dims(1)
            else:
                inputs_embeds = inputs_embeds.expand_dims(1)

        if inputs_embeds is None:
            inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        input_ids = None

        slot_mapping = attn_metadata.slot_mapping
        attn_mask = self.casual_mask.gen_attention_mask(
            is_prefill, positions, query_lens_np, seq_lens_np)
        seq_lens_np = np.array(attn_metadata.seq_lens, dtype=np.int32)
        batch_valid_length = Tensor.from_numpy(seq_lens_np)
        q_seq_lens = Tensor.from_numpy(
            np.array(attn_metadata.query_lens, dtype=np.int32))
        block_tables = attn_metadata.block_tables

        # keep position.ndim to 2, for work on mindspore dynamic shape
        if positions.ndim == 1:
            positions = positions[None]

        model_inputs = (
            input_ids,
            positions,
            key_caches,
            value_caches,
            mutable(is_prefill),
            slot_mapping,
            attn_mask,
            batch_valid_length,
            q_seq_lens,
            block_tables,
            intermediate_tensors,
            inputs_embeds,
        )

        if is_prefill:
            if not self.prefill:
                self.prefill = True
            enable_dynamic_shape(
                self.language_model.model, *model_inputs
            )  # enable dynamic shape once on first prefill step
        else:
            if self.prefill:
                self.prefill = False
                enable_dynamic_shape(
                    self.language_model.model, *model_inputs
                )  # enable dynamic shape once on first decode step

        hidden_states = self.language_model.model(*model_inputs)

        if is_prefill:
            hidden_states = ops.squeeze(hidden_states, 0)
        else:
            hidden_states = ops.squeeze(hidden_states, 1)

        return hidden_states

    def forward(
        self,
        input_ids: mindspore.Tensor,
        positions: mindspore.Tensor,
        kv_caches: list[mindspore.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        **kwargs: object,
    ) -> Union[mindspore.Tensor, IntermediateTensors]:
        """Run forward pass for Qwen2.5-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen2.5-VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            pixel_values: Pixel values to be fed to a model.
                `None` if no images are passed.
            image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in LLM.
                `None` if no images are passed.
            pixel_values_videos: Pixel values of videos to be fed to a model.
                `None` if no videos are passed.
            video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in LLM.
                `None` if no videos are passed.
            second_per_grid_ts: Tensor `(num_videos)` of video time interval (
                in seconds) for each grid along the temporal dimension in the
                3D position IDs. `None` if no videos are passed.
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner from
        # `get_multimodal_embeddings` and `get_input_embeddings`, this
        # condition is only for v0 compatibility.
        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            video_input = self._parse_and_validate_video_input(**kwargs)

            if image_input is None and video_input is None:
                inputs_embeds = None
            else:
                if uses_mrope(self.config):
                    assert positions.ndim == 2 and positions.size(0) == 3, (
                        "multimodal section rotary embedding requires "
                        f"(3, seq_len) positions, but got {positions.size()}")
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids,
                    image_input=image_input,
                    video_input=video_input)
                input_ids = None

        hidden_states = self.run_language_model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors,
            inputs_embeds,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: mindspore.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[mindspore.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def sample(
        self,
        logits: mindspore.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_token = self.language_model.sample(logits, sampling_metadata)
        _pynative_executor.sync()
        return next_token

    def load_weights(self, weights: Iterable[tuple[str, mindspore.Tensor]]):
        self.language_model.load_weights()
