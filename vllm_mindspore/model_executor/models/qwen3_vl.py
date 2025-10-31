# SPDX-License-Identifier: Apache-2.0
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2_5_vl.py
#
# Copyright 2025 Huawei Technologites Co., Ltd
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
"""Inference-only Qwen3-VL model compatible with HuggingFace weights."""
# type: ignore
# isort:skip_file

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Callable, Literal, Optional, TypedDict, Union, Any

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.nn.functional as F
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter, mutable
from mindspore import dtype as mstype
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
from transformers import BatchFeature
from transformers.models.qwen2_vl import (Qwen2VLImageProcessor,
                                          Qwen2VLProcessor)
from transformers.models.qwen2_vl.configuration_qwen2_vl import (Qwen2VLConfig)
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, \
    get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from vllm.distributed import utils as dist_utils
from vllm.logger import init_logger
from vllm.config import get_current_vllm_config
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm_mindspore.model_executor.models.qwen3 import Qwen3ForCausalLM, Qwen3Model

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (ImageItem, ModalityData,
                                    MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs, VideoItem)
from vllm.multimodal.parse import (DictEmbeddingItems, ImageSize,
                                   ModalityDataItems, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import uses_mrope
from vllm.transformers_utils.processor import (
    cached_image_processor_from_config)

from vllm_mindspore.distributed.communication_op import \
    AllGatherFromModelParallelRegion
from vllm_mindspore.model_executor.layers.layernorm import RMSNorm
from vllm_mindspore.model_executor.layers.linear import ColumnParallelLinear, \
    RowParallelLinear, QKVParallelLinear
from vllm_mindspore.model_executor.layers.logits_processor import \
    LogitsProcessor
from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import \
    ParallelLMHead
from vllm_mindspore.model_executor.model_loader.weight_utils import \
    default_weight_loader
from vllm_mindspore.model_executor.models.attention_mask import \
    MultiModalLowerTriangularMask
from vllm_mindspore.model_executor.models.qwen2_5_vl import (
    _qwen2vl_field_config,
    Qwen2_5_VisionAttention,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VLProcessingInfo,
)
from vllm_mindspore.model_executor.models.model_base import NativeModel, \
    AttentionWrapper
from vllm_mindspore.model_executor.models.qwen2 import Qwen2Model
from vllm_mindspore.model_executor.models.utils import PPMissingLayer
from vllm_mindspore.utils import STR_DTYPE_TO_MS_DTYPE
from vllm_mindspore.model_executor.layers.rotary_embedding import _apply_rotary_emb
from .interfaces import (MultiModalEmbeddings, SupportsMultiModal)
from .utils import (WeightsMapper, maybe_prefix, _merge_multimodal_embeddings)

logger = init_logger(__name__)

_ACTIVATION_REGISTRY = {"silu": F.silu}


class Qwen2_5_VLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: ms.Tensor
    """Shape:
    `(num_patches, num_channels * patch_size * patch_size)`
    """

    image_grid_thw: ms.Tensor
    """Shape: `(num_images, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


class Qwen2_5_VLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: ms.Tensor
    """Supported types:
    - list[`ms.Tensor`]: A list of tensors holding all images' features.
        Each tensor holds an image's features.
    - `ms.Tensor`: A tensor holding all images' features
        (concatenation of all images' feature tensors).

    Tensor shape: `(num_image_features, hidden_size)`
    - `num_image_features` varies based on
        the number and resolution of the images.
    - `hidden_size` must match the hidden size of language model backbone.
    """

    image_grid_thw: ms.Tensor
    """Shape: `(num_images, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


class Qwen2_5_VLVideoPixelInputs(TypedDict):
    type: Literal["pixel_values_videos"]
    pixel_values_videos: ms.Tensor
    """Shape:
    `(num_patches,
      num_channels * temporal_patch_size * patch_size * patch_size)`
    """

    video_grid_thw: ms.Tensor
    """Shape: `(num_videos, 3)`

    This should be in `(grid_t, grid_h, grid_w)` format.
    """

    second_per_grid_ts: ms.Tensor
    """
    The video time interval (in seconds) for each grid along the temporal 
    dimension in the 3D position IDs. Returned when `videos` is not `None`.
    """


class Qwen2_5_VLVideoEmbeddingInputs(TypedDict):
    type: Literal["video_embeds"]
    video_embeds: ms.Tensor
    """Supported types:
    - list[`ms.Tensor`]: A list of tensors holding all videos' features.
        Each tensor holds an video's features.
    - `ms.Tensor`: A tensor holding all videos' features
      (concatenation of all videos' feature tensors).

    Tensor shape: `(num_image_features, hidden_size)`
    - `num_image_features` varies based on
        the number and resolution of the videos.
    - `hidden_size` must match the hidden size of language model backbone.
    """

    video_grid_thw: ms.Tensor
    """Shape: `(num_videos, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


Qwen2_5_VLImageInputs = Union[Qwen2_5_VLImagePixelInputs,
                              Qwen2_5_VLImageEmbeddingInputs]

Qwen2_5_VLVideoInputs = Union[Qwen2_5_VLVideoPixelInputs,
                              Qwen2_5_VLVideoEmbeddingInputs]

# For profile run
_MAX_FRAMES_PER_VIDEO = 16
# === Vision Inputs === #


class Qwen2VLProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen2VLConfig)

    def get_hf_processor(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        **kwargs: object,
    ) -> Qwen2VLProcessor:
        return self.ctx.get_hf_processor(
            Qwen2VLProcessor,
            image_processor=self.get_image_processor(
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                size=size,
                # use_fast=kwargs.get("use_fast")),
                use_fast=False)
            **kwargs,
        )

    def _get_image_processor_kwargs(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        **kwargs: object,
    ):
        mm_config = self.ctx.model_config.get_multimodal_config()
        if mm_config.mm_processor_kwargs:
            kwargs.update(mm_config.mm_processor_kwargs)

        if min_pixels is not None:
            kwargs["min_pixels"] = min_pixels

            if size is None:
                size = {"shortest_edge": min_pixels}
            else:
                size["shortest_edge"] = min_pixels

        if max_pixels is not None:
            kwargs["max_pixels"] = max_pixels

            if size is None:
                size = {"longest_edge": max_pixels}
            else:
                size["longest_edge"] = max_pixels

        if size is not None:
            kwargs["size"] = size

        return kwargs

    def get_image_processor(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        **kwargs: object,
    ) -> Qwen2VLImageProcessor:
        return cached_image_processor_from_config(
            self.ctx.model_config,
            **self._get_image_processor_kwargs(min_pixels=min_pixels,
                                               max_pixels=max_pixels,
                                               size=size,
                                               **kwargs),
        )

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None, "video": None}

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 1,
        do_resize: bool = True,
        image_processor: Optional[Qwen2VLImageProcessor],
    ) -> tuple[ImageSize, int]:
        if image_processor is None:
            image_processor = self.get_image_processor()

        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        temporal_patch_size = vision_config.temporal_patch_size

        if do_resize:
            resized_height, resized_width = smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * merge_size,
                min_pixels=image_processor.min_pixels,
                max_pixels=image_processor.max_pixels,
            )
            preprocessed_size = ImageSize(width=resized_width,
                                          height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width,
                                          height=image_height)

        # NOTE: Frames are padded to be divisible by `temporal_patch_size`
        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L294
        padded_num_frames = num_frames + num_frames % temporal_patch_size

        grid_t = max(padded_num_frames // temporal_patch_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (merge_size**2)

        return preprocessed_size, num_vision_tokens

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor: Optional[Qwen2VLImageProcessor],
    ) -> int:
        _, num_image_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            image_processor=image_processor,
        )
        return num_image_tokens

    def get_num_video_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int,
        image_processor: Optional[Qwen2VLImageProcessor],
    ) -> int:
        _, num_video_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            num_frames=num_frames,
            image_processor=image_processor,
        )
        return num_video_tokens

    def get_image_size_with_most_features(self) -> ImageSize:
        max_image_size, _ = self._get_vision_info(
            image_width=9999999,
            image_height=9999999,
            image_processor=None,
        )
        return max_image_size

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
            image_processor=None,
        )

    def _get_max_video_frames(self, max_tokens: int) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        num_frames = 0

        while True:
            next_num_frames = num_frames + 1
            next_max_tokens = self.get_num_video_tokens(
                image_width=target_width,
                image_height=target_height,
                num_frames=next_num_frames,
                image_processor=None,
            )

            if next_max_tokens > max_tokens:
                break

            num_frames = next_num_frames

        return num_frames

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        max_images = mm_counts.get("image", 0)
        max_videos = mm_counts.get("video", 0)

        max_image_tokens = self.get_max_image_tokens() * max_images
        max_total_frames = self._get_max_video_frames(seq_len -
                                                      max_image_tokens)
        max_frames_per_video = min(max_total_frames // max(max_videos, 1),
                                   _MAX_FRAMES_PER_VIDEO)

        return max(max_frames_per_video, 1)

    def get_max_video_tokens(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_video_tokens(
            image_width=target_width,
            image_height=target_height,
            num_frames=self.get_num_frames_with_most_features(
                seq_len, mm_counts),
            image_processor=None,
        )


class Qwen2_5_VLProcessingInfo(Qwen2VLProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen2_5_VLConfig)

    def get_hf_processor(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        fps: Optional[Union[float, list[float]]] = None,
        **kwargs: object,
    ) -> Qwen2_5_VLProcessor:
        if fps is not None:
            kwargs["fps"] = fps

        return self.ctx.get_hf_processor(
            Qwen2_5_VLProcessor,
            image_processor=self.get_image_processor(
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                size=size,
                # use_fast=kwargs.get("use_fast")),
                use_fast=False)
            **kwargs,
        )


class Qwen2VLDummyInputsBuilder(BaseDummyInputsBuilder[Qwen2VLProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        hf_processor = self.info.get_hf_processor()
        image_token: str = hf_processor.image_token
        video_token: str = hf_processor.video_token

        return image_token * num_images + video_token * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        target_num_frames = \
            self.info.get_num_frames_with_most_features(seq_len, mm_counts)

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images),
            "video":
            self._get_dummy_videos(
                width=target_width,
                height=target_height,
                num_frames=target_num_frames,
                num_videos=num_videos,
            )
        }


def _qwen2vl_field_config(hf_inputs: Mapping[str, ms.Tensor]):
    image_grid_thw = hf_inputs.get("image_grid_thw", ms.mint.empty((0, 3)))
    image_grid_sizes = image_grid_thw.prod(-1)

    video_grid_thw = hf_inputs.get("video_grid_thw", ms.mint.empty((0, 3)))
    video_grid_sizes = video_grid_thw.prod(-1)

    return dict(
        pixel_values=MultiModalFieldConfig.flat_from_sizes(
            "image", image_grid_sizes),
        image_embeds=MultiModalFieldConfig.flat_from_sizes(
            "image", image_grid_sizes),
        image_grid_thw=MultiModalFieldConfig.batched("image"),
        pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
            "video", video_grid_sizes),
        video_embeds=MultiModalFieldConfig.flat_from_sizes(
            "video", video_grid_sizes),
        video_grid_thw=MultiModalFieldConfig.batched("video"),
    )


class Qwen2VLMultiModalDataParser(MultiModalDataParser):

    def _parse_image_data(
        self,
        data: Union[dict[str, ms.Tensor], ModalityData[ImageItem]],
    ) -> Optional[ModalityDataItems[Any, Any]]:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="image",
                required_fields={"image_embeds", "image_grid_thw"},
                fields_factory=_qwen2vl_field_config,
            )

        return super()._parse_image_data(data)

    def _parse_video_data(
        self,
        data: Union[dict[str, ms.Tensor], ModalityData[VideoItem]],
    ) -> Optional[ModalityDataItems[Any, Any]]:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="video",
                required_fields={"video_embeds", "video_grid_thw"},
                fields_factory=_qwen2vl_field_config,
            )

        return super()._parse_video_data(data)


class Qwen2VLMultiModalProcessor(BaseMultiModalProcessor[Qwen2VLProcessingInfo]
                                 ):

    def _get_data_parser(self) -> MultiModalDataParser:
        return Qwen2VLMultiModalDataParser()

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        return self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**mm_kwargs),
            dict(text=prompt, **mm_data),
            self.info._get_image_processor_kwargs(**mm_kwargs),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
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
            assert isinstance(grid_thw, ms.Tensor)

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

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _qwen2vl_field_config(hf_inputs)


class _Qwen2VLMultiModalProcessor(Qwen2VLMultiModalProcessor):

    def _get_prompt_updates(
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
            assert isinstance(grid_thw, ms.Tensor)

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


# === Vision Encoder === #


class Qwen3_VisionMLP(nn.Cell):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = False,
        act_fn: Callable[[ms.Tensor], ms.Tensor] = F.silu,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.linear_fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_fc1",
            params_dtype=ms.bfloat16
        )
        self.linear_fc2 = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_fc2",
            params_dtype=ms.bfloat16
        )
        self.act_fn = act_fn

    def construct(self, x: ms.Tensor):
        mlp_output = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return mlp_output


def apply_rotary_pos_emb_flashatt(
        q: ms.Tensor, k: ms.Tensor, cos: ms.Tensor,
        sin: ms.Tensor) -> tuple[ms.Tensor, ms.Tensor]:
    q_embed = ops.rotary_position_embedding(q.float(), cos, sin).type_as(q)
    k_embed = ops.rotary_position_embedding(k.float(), cos, sin).type_as(k)
    return q_embed, k_embed


class Qwen3_VisionAttention(Qwen2_5_VisionAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flash_attention_score = FlashAttentionScore(head_num=self.num_attention_heads_per_partition,
                                                         scale_value=1 / math.sqrt(self.hidden_size_per_attention_head),
                                                         input_layout="TH")
    def construct(
            self,
            x: Tensor,
            batch_valid_length: Tensor,
            position_embeddings: tuple[ms.Tensor, ms.Tensor],
            q_seq_lens: Tensor
    ) -> Tensor:
        seq_length = x.shape[0]
        qkv, _ = self.qkv(x)
        q, k, v = mint.split(
            qkv, (self.num_attention_heads_per_partition * self.head_dim,
                  self.num_attention_heads_per_partition * self.head_dim,
                  self.num_attention_heads_per_partition * self.head_dim), -1)

        # q/k reshape to BSND
        # q = q.reshape(1, seq_length, self.num_attention_heads_per_partition,
        #               self.hidden_size_per_attention_head)
        # k = k.reshape(1, seq_length, self.num_attention_heads_per_partition,
        #               self.hidden_size_per_attention_head)
        q = q.reshape(seq_length, self.num_attention_heads_per_partition,
                      self.hidden_size_per_attention_head)
        k = k.reshape(seq_length, self.num_attention_heads_per_partition,
                      self.hidden_size_per_attention_head)

        cos, sin = position_embeddings
        origin_dtype = q.dtype
        # q, k = ms_custom_ops.apply_rotary_pos_emb_ext(q.astype(ms.float32),
        #                                               k.astype(ms.float32),
        #                                               cos, sin, "BSND", "half")
        q = _apply_rotary_emb(q, cos, sin, True)
        k = _apply_rotary_emb(k, cos, sin, True)

        # q/k reshape to TH
        q = q.astype(origin_dtype)
        k = k.astype(origin_dtype)
        q = q.reshape(seq_length, self.num_attention_heads_per_partition * self.hidden_size_per_attention_head)
        k = k.reshape(seq_length, self.num_attention_heads_per_partition * self.hidden_size_per_attention_head)
        v = v.reshape(seq_length, self.num_attention_heads_per_partition * self.hidden_size_per_attention_head)

        _, _, _, context_layer = self.flash_attention_score(
            q,
            k,
            v,
            None,
            None,
            None,
            None,
            None,
            batch_valid_length,
            q_seq_lens,
        )
        output, _ = self.proj(context_layer)
        return output


class Qwen3_VisionBlock(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[ms.Tensor], ms.Tensor] = F.silu,
        norm_layer: Callable[[int], nn.Cell] | None = None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(mint.nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Qwen3_VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = Qwen3_VisionMLP(
            dim,
            mlp_hidden_dim,
            act_fn=act_fn,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def construct(
        self,
        x: ms.Tensor,
        batch_valid_length: ms.Tensor,
        position_embeddings: ms.Tensor,
        q_seq_lens: list[int] | None = None,  # Only used for xFormers
    ) -> ms.Tensor:
        x = x + self.attn(
            self.norm1(x),
            batch_valid_length,
            position_embeddings,
            q_seq_lens
        )

        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3_VisionPatchEmbed(nn.Cell):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        # self.dtype = get_current_vllm_config().model_config.dtype
        self.dtype = ms.bfloat16

        # Use Dense layer instead of Conv3d for MindSpore compatibility
        self.proj = ms.nn.Dense(temporal_patch_size * patch_size * patch_size * in_channels,
                                hidden_size,
                                has_bias=True,
                                dtype=self.dtype)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.proj(x)
        return x


class Qwen3_VisionPatchMerger(nn.Cell):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable[[int], nn.Cell] | None = None,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)

        self.use_postshuffle_norm = use_postshuffle_norm
        if self.use_postshuffle_norm:
            context_dim = self.hidden_size

        if norm_layer is None:
            norm_layer = partial(mint.nn.LayerNorm, eps=1e-6)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.ln_q = norm_layer(
            self.hidden_size if use_postshuffle_norm else context_dim
        )
        self.mlp = nn.CellList(
            [
                ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=True,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp.0",
                    params_dtype=ms.bfloat16
                ),
                nn.GELU(),
                RowParallelLinear(
                    self.hidden_size,
                    d_model,
                    bias=True,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp.2",
                    params_dtype=ms.bfloat16
                ),
            ]
        )

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        if self.use_postshuffle_norm:
            x = self.ln_q(x.view(-1, self.hidden_size))
        else:
            x = self.ln_q(x).view(-1, self.hidden_size)

        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x_parallel, _ = mlp_fc1(x)
        x_parallel = mlp_act(x_parallel)
        out, _ = mlp_fc2(x_parallel)
        return out


class Qwen3_VisionTransformer(nn.Cell):
    def __init__(
        self,
        vision_config,
        norm_eps: float = 1e-6,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vision_config = vision_config
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.image_size = vision_config.image_size
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.num_grid_per_side = self.image_size // self.patch_size
        self.apply_vit_abs_pos_embed = vision_config.apply_vit_abs_pos_embed
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes
        # self.dtype = get_current_vllm_config().model_config.dtype
        self.dtype = ms.bfloat16

        self.patch_embed = Qwen3_VisionPatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )

        # vit pos embeding, TODO: spatial_patch_size vs patch_size
        if self.apply_vit_abs_pos_embed:
            self.pos_embed = mint.nn.Embedding(self.num_grid_per_side**2, self.hidden_size, dtype=self.dtype)
        else:
            self.pos_embed = Parameter(
                mint.empty([1, self.num_grid_per_side**2, self.hidden_size], dtype=self.dtype)
            )

        norm_layer = partial(mint.nn.LayerNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.CellList(
            [
                Qwen3_VisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    act_fn=_ACTIVATION_REGISTRY[vision_config.hidden_act],
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(vision_config.depth)
            ]
        )
        self.merger = Qwen3_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            norm_layer=norm_layer,
            spatial_merge_size=self.spatial_merge_size,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
        )
        if self.deepstack_visual_indexes is not None:
            self.merger_list = nn.CellList(
                [
                    Qwen3_VisionPatchMerger(
                        d_model=vision_config.out_hidden_size,
                        context_dim=self.hidden_size,
                        spatial_merge_size=self.spatial_merge_size,
                        use_postshuffle_norm=True,
                        norm_layer=norm_layer,
                        quant_config=quant_config,
                        prefix=f"{prefix}.merger_list.{layer_idx}",
                    )
                    for layer_idx in range(len(self.deepstack_visual_indexes))
                ]
            )

    # @property
    # def dtype(self) -> ms.dtype:
    #     return self.patch_embed.proj.weight.dtype

    def construct(
        self,
        x: ms.Tensor,
        batch_valid_length: ms.Tensor,
        q_seq_lens: ms.Tensor,
        rotary_pos_emb: ms.Tensor,
        pos_embeds: ms.Tensor,
    ) -> ms.Tensor:
        hidden_states = x.astype(self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        if self.apply_vit_abs_pos_embed:
            hidden_states = hidden_states + pos_embeds
        # rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # hidden_states = hidden_states.unsqueeze(1)
        seq_len, _ = x.shape
        rotary_pos_emb = rotary_pos_emb.astype(hidden_states.dtype)
        # rotary_pos_emb = rotary_pos_emb.reshape(1, seq_len, 1, -1)
        emb = rotary_pos_emb
        # emb = mint.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (mint.cos(emb), mint.sin(emb))

        hidden_states_list = []
        deepstack_visual_indexes = self.deepstack_visual_indexes

        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                batch_valid_length,
                position_embeddings,
                q_seq_lens
            )
            if (
                deepstack_visual_indexes is not None
                and layer_num in deepstack_visual_indexes
            ):
                hidden_states_list.append(hidden_states)

        hidden_states = self.merger(hidden_states)

        # processing deepstack
        if deepstack_visual_indexes is not None:
            processed_hidden_states_list = [hidden_states]
            for idx, x in enumerate(hidden_states_list):
                x = self.merger_list[idx](x)
                processed_hidden_states_list.append(x)
            # we cat the original visual features and deepstack features
            # along the feature dim
            hidden_states = mint.cat(
                processed_hidden_states_list, dim=1
            )  # [seq_len, hidden_size * (1 + depth_of_deepstack)]

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, Tensor]],
                     params_dict: dict[str, Parameter]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("attn.qkv.", "attn.q.", "q"),
            ("attn.qkv.", "attn.k.", "k"),
            ("attn.qkv.", "attn.v.", "v"),
        ]
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                if "patch_embed.proj.weight" in name:
                    loaded_weight = loaded_weight[:]
                    loaded_weight = loaded_weight.reshape(loaded_weight.shape[0],
                                                            -1)
                    param.set_data(ms.Tensor(loaded_weight, dtype=param.dtype))
                else:
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def set_model_inputs(self):
        x_dtype = get_current_vllm_config().model_config.dtype
        dyn_x = ms.Tensor(shape=[None, None], dtype=x_dtype) # 1932 * 1092
        dyn_batch_valid_length = ms.Tensor(shape=[None], dtype=ms.int32)
        dyn_q_seq_lens = ms.Tensor(shape=[None], dtype=ms.int32)
        dyn_rotary_pos_emb = ms.Tensor(shape=[None, None], dtype=ms.float32)
        dyn_pos_emb = ms.Tensor(shape=[None, None], dtype=x_dtype)

        self.set_inputs(
            dyn_x,
            dyn_batch_valid_length,
            dyn_q_seq_lens,
            dyn_rotary_pos_emb,
            dyn_pos_emb
        )


class Qwen3LLMModel(Qwen3Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", deepstack_layers: int = 0):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

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
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[Tensor] = None,
        deepstack_input_embeds: Optional[Mapping[str, Tensor]] = None,
    ) -> ms.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        # for layer_idx, layer in enumerate(
        #     self.layers[self.start_layer : self.end_layer]
        # ):
        #     layer_idx = layer_idx + self.start_layer

        #     hidden_states, residual = layer(
        #         positions,
        #         hidden_states,
        #         residual,
        #     )
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, key_caches[i - self.start_layer],
                value_caches[i - self.start_layer], slot_mapping, attn_mask,
                batch_valid_length, q_seq_lens, block_tables, residual,
                None, None, None, None)

            if deepstack_input_embeds is not None and i in range(self.deepstack_layers):
                hidden_states = (
                    hidden_states
                    + deepstack_input_embeds[f"deepstack_input_embeds_{i}"]
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3LLMForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", deepstack_layers: int = 0):
        super(Qwen3ForCausalLM, self).__init__(vllm_config=vllm_config)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3LLMModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"),
            deepstack_layers=deepstack_layers
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, quant_config=quant_config
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )


class Qwen2_5_VLMultiModalProcessor(_Qwen2VLMultiModalProcessor):

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            **super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs),
            second_per_grid_ts=MultiModalFieldConfig.batched("video"),
        )


class Qwen2_5_VLDummyInputsBuilder(
        BaseDummyInputsBuilder[Qwen2VLProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        hf_processor = self.info.get_hf_processor()
        image_token: str = hf_processor.image_token
        video_token: str = hf_processor.video_token

        return image_token * num_images + video_token * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        target_num_frames = \
            self.info.get_num_frames_with_most_features(seq_len, mm_counts)

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images),
            "video":
            self._get_dummy_videos(
                width=target_width,
                height=target_height,
                num_frames=target_num_frames,
                num_videos=num_videos,
            )
        }


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder)
class Qwen3_VLForConditionalGeneration(NativeModel, SupportsMultiModal):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }
    # LoRA specific attributes
    supported_lora_modules = [
        # language model
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",  # Same name with vision encoder
        # vision tower
        "qkv",
        "gate_proj",
        "up_proj",
        "attn.proj",  # Distinguish patch_embed.proj
        "fc1",
        "fc2",
        # projector
        "mlp.0",
        "mlp.2"
    ]

    embedding_modules = {}  # type: ignore[var-annotated]
    embedding_padding_modules = []  # type: ignore[var-annotated]

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "lm_head.": "language_model.lm_head.",
        "model.": "language_model.model.",
    })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        self.visual = Qwen3_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )

        # Now, in order to meet the requirements of vit ->aclnn fa and
        # llm ->atb fa, the only way is to block atb's fa before entering
        # vit and let it go through aclnn's fa, and after exiting vit,
        # in the unset environment variable. If vit also follows pynative mode,
        # the environment variables will not take effect. The solution is
        # to use different primitives for different kernels in the backend.
        self.visual.construct = ms.jit(function=self.visual, jit_level='O0')
        self.visual.set_model_inputs()
        logger.info(
            "PyNative is not available for vit, vit is always graph mode.")

        self.use_deepstack = hasattr(
            config.vision_config, "deepstack_visual_indexes"
        )
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.language_model = Qwen3LLMForCausalLM(
            vllm_config=vllm_config.with_hf_config(
                config.text_config, architectures=["Qwen3ForCausalLM"]
            ),
            prefix=maybe_prefix(prefix, "language_model"),
            deepstack_layers=self.deepstack_num_level
        )
        self.model = self.language_model.model
        self.text_config = config.text_config
        self.lm_head = self.language_model.lm_head
        self.common_preprocess(vllm_config, prefix)

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # register buffer for deepstack
        self.deepstack_input_embeds = (
            [
                mint.zeros(
                    (vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.hidden_size),
                    dtype=self.model_config.dtype
                )
                for _ in range(self.deepstack_num_level)
            ]
            if self.use_deepstack
            else None
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level
        head_dim = (self.vision_config.hidden_size // 
                   self.vision_config.num_heads)
        self.rotary_pos_emb_full = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

    def common_preprocess(self, vllm_config, prefix=""):
        self.set_modules({
            "thinker.visual": self.visual,
            "thinker.model": self.language_model.model,
            "thinker.lm_head": self.language_model.lm_head
        })
        self.casual_mask = MultiModalLowerTriangularMask(
            dtype=self.model_config.dtype,
            max_model_len=self.model_config.max_model_len)
        self.kv_caches = [
            AttentionWrapper() for i in range(self.text_config.num_hidden_layers)
        ]

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(self.text_config.num_hidden_layers):
            compilation_config.static_forward_context[str(
                i)] = self.kv_caches[i]

    def _get_deepstack_input_embeds(self, num_tokens: int) -> IntermediateTensors:
        # get deepstack_input_embeds from buffer, and clear the buffer
        return IntermediateTensors(
            {
                f"deepstack_input_embeds_{idx}": self.deepstack_input_embeds[idx][
                    :num_tokens
                ]
                for idx in range(self.deepstack_num_level)
            }
        )

    def _set_deepstack_input_embeds(self, deepstack_input_embeds: ms.Tensor) -> None:
        # set deepstack_input_embeds to buffer
        num_tokens = deepstack_input_embeds.shape[1]
        if num_tokens > self.deepstack_input_embeds[0].shape[0]:
            self.deepstack_input_embeds = [
                mint.zeros(
                    (num_tokens,
                    self.config.text_config.hidden_size),
                    dtype=self.deepstack_input_embeds[0].dtype,
                )
                for _ in range(self.deepstack_num_level)
            ]
        for idx in range(self.deepstack_num_level):
            self.deepstack_input_embeds[idx][:num_tokens] = deepstack_input_embeds[idx]

    def _clear_deepstack_input_embeds(self, num_tokens: int) -> None:
        # clear deepstack_input_embeds in buffer
        if num_tokens > 0:
            for idx in range(self.deepstack_num_level):
                self.deepstack_input_embeds[idx][:num_tokens] = 0

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
            if (
                input_key in ("pixel_values_videos", "video_embeds")
                and "video" not in mm_input_by_modality
            ):
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(
                    **kwargs
                )
        return mm_input_by_modality

    def get_language_model(self) -> nn.Cell:
        return self.language_model

    def get_multimodal_embeddings(
        self, **kwargs: object
    ) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[ms.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += tuple(video_embeddings)
        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: ms.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        # *,
        # is_multimodal: ms.Tensor | None = None,
        # handle_oov_mm_token: bool = False,
    ) -> ms.Tensor:
        # inputs_embeds = self._get_text_embeddings(
        #     input_ids,
        #     self.language_model.get_input_embeddings,
        #     is_multimodal=is_multimodal,
        #     handle_oov_mm_token=handle_oov_mm_token,
        # )
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        placeholder_token_id = [self.config.image_token_id,
                                self.config.video_token_id,
                                self.config.audio_token_id]
        is_multimodal = ms.numpy.isin(input_ids, placeholder_token_id)

        deepstack_input_embeds = None
        # TODO (ywang96): support overlapping modalitiy embeddings so that
        # `use_audio_in_video` will work on V1.
        # split the feat dim to obtain multi-scale visual feature
        has_vision_embeddings = [
            embeddings.shape[-1] != self.config.text_config.hidden_size
            for embeddings in multimodal_embeddings
        ]
        if self.visual.deepstack_visual_indexes is not None and any(
            has_vision_embeddings
        ):
            multiscale_len = len(self.visual.deepstack_visual_indexes)
            multimodal_embeddings_multiscale = []
            is_vision = mint.zeros_like(is_multimodal)
            mm_positions = mint.nonzero(is_multimodal, as_tuple=True)[0]
            mm_position_idx = 0
            for index, embeddings in enumerate(multimodal_embeddings):
                num_tokens = embeddings.shape[0]
                current_positions = mm_positions[
                    mm_position_idx : mm_position_idx + num_tokens
                ]

                # Vision embeddings
                if embeddings.shape[-1] != self.config.text_config.hidden_size:
                    visual_dim = embeddings.shape[-1] // (multiscale_len + 1)
                    multi_dim = visual_dim * multiscale_len
                    embeddings_main, embeddings_multiscale = mint.split(
                        embeddings, [visual_dim, multi_dim], dim=-1
                    )
                    multimodal_embeddings[index] = embeddings_main
                    multimodal_embeddings_multiscale.append(embeddings_multiscale)
                    is_vision[current_positions] = True

                # Audio embeddings
                else:
                    is_vision[current_positions] = False

                mm_position_idx += num_tokens

            deepstack_input_embeds = inputs_embeds.new_zeros(
                inputs_embeds.shape[0], multiscale_len * inputs_embeds.shape[1])
            deepstack_input_embeds = _merge_multimodal_embeddings(
                inputs_embeds=deepstack_input_embeds,
                multimodal_embeddings=multimodal_embeddings_multiscale,
                is_multimodal=is_vision,
            )
            deepstack_input_embeds = (
                deepstack_input_embeds.view(
                    inputs_embeds.shape[0], multiscale_len, visual_dim
                )
                .permute(1, 0, 2)
            )
            self._set_deepstack_input_embeds(deepstack_input_embeds)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def forward(
        self,
        input_ids: ms.Tensor,
        positions: ms.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: ms.Tensor | None = None,
        **kwargs: object,
    ) -> ms.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        if (
            self.use_deepstack
            and inputs_embeds is not None
            and get_pp_group().is_first_rank
        ):
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.shape[0]
            )
        else:
            deepstack_input_embeds = None


        hidden_states = self.exec_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            # args for deepstack
            deepstack_input_embeds=deepstack_input_embeds,
        )

        if inputs_embeds is not None and get_pp_group().is_first_rank:
            self._clear_deepstack_input_embeds(inputs_embeds.shape[0])

        return hidden_states

    def compute_logits(
        self,
        hidden_states: ms.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> ms.Tensor | None:
        return self.language_model.compute_logits(hidden_states, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, ms.Tensor]]) -> set[str]:
        params_dict = self.get_params_dict()
        loaded_param = set()
        visual_load = set()
        text_load = set()
        for name, weight in weights:
            if "thinker.visual." in name:
                visual_load.update(
                    self.visual.load_weights([(name, weight)], params_dict))
            elif "thinker.model." in name:
                text_load.update(
                    self.model.load_weights([(name, weight)], params_dict))
            else:
                # Handle other weights
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, weight)
                    loaded_param.add(name)
        loaded_param.update(visual_load)
        loaded_param.update(text_load)
        return None #loaded_param talker not supported yet

    def rot_pos_emb(self, grid_thw: ms.Tensor) -> ms.Tensor:
        spatial_merge_size = self.vision_config.spatial_merge_size
        pos_ids = []
        for t, h, w in grid_thw:
            t, h, w = int(t.item()), int(h.item()), int(w.item())
            hpos_ids = mint.arange(h).unsqueeze(1).expand((-1, w))
            wpos_ids = mint.arange(w).unsqueeze(0).expand((h, -1))

            hpos_ids = hpos_ids.reshape(
                h // spatial_merge_size,
                spatial_merge_size,
                w // spatial_merge_size,
                spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            wpos_ids = wpos_ids.reshape(
                h // spatial_merge_size,
                spatial_merge_size,
                w // spatial_merge_size,
                spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            pos_ids.append(
                mint.tile(mint.stack([hpos_ids, wpos_ids], dim=-1), (t, 1)))
        pos_ids = mint.cat(pos_ids, dim=0)
        max_grid_size = int(grid_thw[:, 1:].max().item())
        rotary_pos_emb_full = self.rotary_pos_emb_full(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> ms.Tensor:
        num_grid_per_side = self.visual.num_grid_per_side
        m_size = self.visual.spatial_merge_size
        hidden_dim = self.visual.pos_embed.embedding_dim

        outputs = []
        for t, h, w in grid_thw:
            h_idxs = mint.linspace(
                0, num_grid_per_side - 1, h, dtype=ms.float32
            )
            w_idxs = mint.linspace(
                0, num_grid_per_side - 1, w, dtype=ms.float32
            )

            h_floor = h_idxs.astype(ms.int64)
            w_floor = w_idxs.astype(ms.int64)
            h_ceil = mint.clamp(h_floor + 1, 0, num_grid_per_side - 1)
            w_ceil = mint.clamp(w_floor + 1, 0, num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            # Create meshgrid view for all h, w vars
            dh_grid, dw_grid = mint.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = mint.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil_grid, w_ceil_grid = mint.meshgrid(h_ceil, w_ceil, indexing="ij")
            h_floor_grid_idx = h_floor_grid * num_grid_per_side
            h_ceil_grid_idx = h_ceil_grid * num_grid_per_side

            # original computation of weights
            # w00 = (1 - dh_grid) * (1 - dw_grid)
            # w01 = (1 - dh_grid) * dw_grid
            # w10 = dh_grid * (1 - dw_grid)
            # w11 = dh_grid * dw_grid
            # we reuse w11 here to avoid duplicate
            # dh_grid * dw_grid computation
            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - dw_grid + w11

            idx00 = h_floor_grid_idx + w_floor_grid
            idx01 = h_floor_grid_idx + w_ceil_grid
            idx10 = h_ceil_grid_idx + w_floor_grid
            idx11 = h_ceil_grid_idx + w_ceil_grid

            indices = mint.stack([idx00, idx01, idx10, idx11], dim=0).reshape(4, -1)
            weights = mint.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
            weights = weights.astype(self.visual.dtype)

            embeds = self.visual.pos_embed(indices)
            weighted_embeds = embeds * weights
            p0, p1, p2, p3 = weighted_embeds.unbind(dim=0)
            combined = p0 + p1 + p2 + p3

            combined = combined.view(h * w, hidden_dim)
            repeated = combined.unsqueeze(0).expand(t, -1, -1)
            repeated = repeated.view(
                t, h // m_size, m_size, w // m_size, m_size, hidden_dim
            )
            repeated = repeated.permute(0, 1, 3, 2, 4, 5).reshape(-1, hidden_dim)
            outputs.append(repeated)

        return mint.cat(outputs, dim=0)

    def set_model_inputs(self,
                         input_ids=None,
                         position_ids=None,
                         intermediate_tensors=None,
                         inputs_embeds=None):
        if input_ids is None:
            dyn_input_ids = None
        else:
            dyn_input_ids = ms.Tensor(shape=[None] * input_ids.ndim,
                                      dtype=mstype.int32)

        if position_ids is None:
            dyn_position_ids = None
        else:
            dyn_position_ids = ms.Tensor(shape=[None] * position_ids.ndim,
                                         dtype=mstype.int32)

        if inputs_embeds is None:
            dyn_inputs_embeds = None
        else:
            dyn_inputs_embeds = ms.Tensor(shape=[None] * inputs_embeds.ndim,
                                          dtype=inputs_embeds.dtype)

        if intermediate_tensors is None:
            dyn_intermediate_tensors = None
        else:
            dyn_intermediate_tensors = ms.Tensor(
                shape=[None] * intermediate_tensors.ndim,
                dtype=intermediate_tensors.dtype)

        block_size = self.cache_config.block_size
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()
        kv_cache_shape = (None, block_size, num_kv_heads, head_size)

        kv_cache_dtype = (self.model_config.dtype
                          if self.cache_config.cache_dtype == "auto" else
                          self.cache_config.cache_dtype)
        if kv_cache_dtype in STR_DTYPE_TO_MS_DTYPE:
            kv_cache_dtype = STR_DTYPE_TO_MS_DTYPE[kv_cache_dtype]

        num_layers = self.model_config.get_num_layers(self.parallel_config)

        dyn_key_cache = Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype)
        dyn_value_cache = Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype)
        dyn_key_caches = mutable([dyn_key_cache for _ in range(num_layers)])
        dyn_value_caches = mutable(
            [dyn_value_cache for _ in range(num_layers)])

        dyn_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_attention_mask = Tensor(shape=[None, None],
                                        dtype=self.model_config.dtype)
        dyn_batch_valid_length = Tensor(shape=[None], dtype=mstype.int32)
        dyn_q_seq_lens = Tensor(shape=[None], dtype=mstype.int32)
        dyn_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dyn_deepstack_input_embeds = Tensor(shape=[None, None],
                                            dtype=self.model_config.dtype)

        self.ready_model.set_inputs(
            dyn_input_ids, dyn_position_ids, dyn_key_caches,
            dyn_value_caches, dyn_slot_mapping, dynamic_attention_mask,
            dyn_batch_valid_length, dyn_q_seq_lens, dyn_block_tables,
            dyn_intermediate_tensors, dyn_inputs_embeds, dyn_deepstack_input_embeds)

        dynamic_hidden_states = Tensor(shape=[None, None],
                                       dtype=self.model_config.dtype)
        self.ready_lm_head.set_inputs(dynamic_hidden_states)

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                        name: str) -> ms.Tensor:
        if not isinstance(mm_input, (ms.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                             f"Got type: {type(mm_input)}")
        if isinstance(mm_input, ms.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(f"{name} should be 2D or batched 3D tensor. "
                                 f"Got ndim: {mm_input.ndim} "
                                 f"(shape={mm_input.shape})")
            return mint.concat(list(mm_input))
        else:
            return mint.concat(mm_input)

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

            if not isinstance(pixel_values, (ms.Tensor, list)):
                raise ValueError("Incorrect type of image pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return Qwen2_5_VLImagePixelInputs(type="pixel_values",
                                              pixel_values=pixel_values,
                                              image_grid_thw=image_grid_thw)

        if image_embeds is not None:
            image_embeds = self._validate_and_reshape_mm_tensor(
                image_embeds, "image embeds")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(image_embeds, ms.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            return Qwen2_5_VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw)

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

            if not isinstance(video_embeds, ms.Tensor):
                raise ValueError("Incorrect type of video embeddings. "
                                 f"Got type: {type(video_embeds)}")
            return Qwen2_5_VLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw)

        return None

    def _process_image_input(
        self, image_input
    ) -> tuple[Tensor, ...]:
        if image_input["type"] == "image_embeds":
            return image_input["image_embeds"].type(self.visual.dtype)

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        pos_emb = self.fast_pos_embed_interpolate(grid_thw.tolist())
        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        grid_thw_1 = grid_thw.index_select(1, ms.Tensor([1])).reshape(-1)
        grid_thw_2 = grid_thw.index_select(1, ms.Tensor([2])).reshape(-1)
        grid_thw_0 = grid_thw.index_select(1, ms.Tensor([0])).reshape(-1)
        batch_valid_length = mint.repeat_interleave(grid_thw_1 * grid_thw_2, grid_thw_0).astype(ms.int32)
        image_embeds = self.visual(pixel_values, batch_valid_length, batch_valid_length,
                                   rotary_pos_emb, pos_emb)
        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return image_embeds.split(sizes.tolist())

    def _process_video_input(
            self, video_input
    ) -> tuple[Tensor, ...]:
        if video_input["type"] == "video_embeds":
            return video_input["video_embeds"].type(self.visual.dtype)

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        pos_emb = self.fast_pos_embed_interpolate(grid_thw.tolist())
        pixel_values = video_input["pixel_values_video"].type(self.visual.dtype)
        grid_thw_1 = grid_thw.index_select(1, ms.Tensor([1])).reshape(-1)
        grid_thw_2 = grid_thw.index_select(1, ms.Tensor([2])).reshape(-1)
        grid_thw_0 = grid_thw.index_select(1, ms.Tensor([0])).reshape(-1)
        batch_valid_length = mint.repeat_interleave(grid_thw_1 * grid_thw_2, grid_thw_0).astype(ms.int32)
        video_embeds = self.visual(pixel_values, batch_valid_length, batch_valid_length,
                                   rotary_pos_emb, pos_emb)
        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return video_embeds.split(sizes.tolist())


    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values",
                             "image_embeds") and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(
                    **kwargs)
            if input_key in ("pixel_values_videos",
                             "video_embeds") and "videos" not in modalities:
                modalities["videos"] = self._parse_and_validate_video_input(
                    **kwargs)
        return modalities
