# SPDX-License-Identifier: Apache-2.0
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/phi4mm.py
#
# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Copyright 2025 The vLLM Team.
# Copyright 2025 Microsoft Corporation.
# All rights reserved.
#
# This file is adapted from the vLLM implementation of the Phi-4 multimodal model.
# It has been modified to support the MindSpore/vLLM-MindSpore inference framework.
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
"""Inference-only Phi-4-multimodal-instruct model compatible with HuggingFace weights."""
# type: ignore
# isort:skip_file

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, TypeAlias, TypedDict, Union, List,  Optional

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
import mindspore.mint as mint
import mindspore.mint.nn.functional as F
from vllm_mindspore.model_executor.models.model_base import AttentionWrapper, NativeModel


from mindone.transformers import BatchFeature, ProcessorMixin, SequenceFeatureExtractor
from transformers import PretrainedConfig, SiglipVisionConfig
from mindone.transformers.utils import TensorType

from mindone.transformers.models.phi4_multimodal.processing_phi4_multimodal import Phi4MMProcessor

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group

from vllm_mindspore.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead)

from vllm_mindspore.model_executor.models.llama import LlamaModel,LlamaForCausalLM
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm_mindspore.multimodal.inputs import NestedTensors


from vllm.multimodal.parse import (AudioProcessorItems, ImageEmbeddingItems,
                                   ImageProcessorItems, ImageSize,
                                   MultiModalDataItems, MultiModalDataParser)

from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement, ResolvedPromptUpdate)

from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils import is_list_of

from .idefics2_vision_model import Idefics2VisionTransformer
from vllm_mindspore.model_executor.models.interfaces import (
    MultiModalEmbeddings, SupportsMultiModal)

from vllm.model_executor.models.interfaces import SupportsLoRA

from .phi4mm_audio import AudioEmbedding
from .utils import (
    WeightsMapper, 
    flatten_bn, 
    maybe_prefix,
    merge_multimodal_embeddings)

# <|endoftext10|> (see vocab.json in hf model)
_IMAGE_PLACEHOLDER_TOKEN_ID = 200010
# <|endoftext11|>
_AUDIO_PLACEHOLDER_TOKEN_ID = 200011

_AUDIO_MAX_SOUNDFILE_SIZE = 241_000

SIGLIP_NAME = "siglip-so400m-patch14-448"
VISION_ENCODER_TO_PROCESSING_CONFIG = {
    'siglip-so400m-patch14-448': {
        'vit_image_size': 448,
        'vit_patch_size': 14,
        'token_compression_factor': 2,
    },
}

def _get_padding_size(orig_width: int, orig_height: int, target_height: int,
                      target_width: int):
    ratio_width = target_width / orig_width
    ratio_height = target_height / orig_height

    if ratio_width < ratio_height:
        padding_width = 0
        padding_height = target_height - int(orig_height * ratio_width)
    else:
        padding_width = target_width - int(orig_width * ratio_height)
        padding_height = 0
    return padding_height, padding_width

def get_navit_vision_model(layer_idx: int = -1, **kwargs):
    vision_config = {
        "hidden_size": 1152,
        "image_size": 448,
        "intermediate_size": 4304,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14,
    }

    model_config = SiglipVisionConfig(**vision_config, **kwargs)
    if layer_idx < 0:
        num_hidden_layers = model_config.num_hidden_layers \
            + layer_idx + 1
    else:
        num_hidden_layers = layer_idx + 1

    vision_model = Idefics2VisionTransformer(
        config=model_config,
        require_post_norm=False,
        num_hidden_layers_override=num_hidden_layers,
    )

    return vision_model


class Phi4MMImageEncoder(nn.Cell):
    """Image embedding."""

    def __init__(self,
                 config: PretrainedConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 model_dir: str = "") -> None:
        super().__init__()

         # n_embed or hidden_size
        hidden_size = config.n_embd if hasattr(
            config, 'n_embd') else config.hidden_size

        if isinstance(config.img_processor, dict):
            self.layer_idx = config.img_processor.get('layer_idx', -2)
            self.type_feature = config.img_processor.get('type_feature', 'patch')
        else:
            self.layer_idx = -2
            self.type_feature = 'patch'

        self.img_processor = get_navit_vision_model(layer_idx=self.layer_idx)

        pe_weight = self.img_processor.embeddings.position_embedding.weight
        
        L, D = pe_weight.shape
        H = int(math.sqrt(L))
        assert H**2 == L, f'position embedding size {L} is not square'
        if H % 2 != 0:
            self.img_processor_padding = mint.nn.ReflectionPad2d((0, 1, 0, 1))
            H += 1
        image_dim_out = D
        self.num_img_tokens = (H // 2)**2
        self.base_feat_height_target = H

        self.image_dim_out = image_dim_out
        self.img_sizes = None
        self.image_attention_mask = None

        self.use_hd_transform = True
        self.with_learnable_separator = True
        self.hd_transform_order = "sub_glb"
        self.freeze_img_processor = False
        self.crop_size = 448

        self.image_token_compression_cls = 'avg_pool_2d'
        self.image_token_compression = mint.nn.AvgPool2d(kernel_size=2, stride=2)
        self.base_feat_height_reduction = 1
        self.base_feat_height_target = self.base_feat_height_target // 2

        assert self.use_hd_transform == self.with_learnable_separator, \
            'use_hd_transform and with_learnable_separator should have same value'
        assert self.use_hd_transform, 'learnable separator is only for hd transform'

        self.glb_GN = ms.Parameter(mint.zeros((1, 1, self.image_dim_out * self.base_feat_height_reduction**2), dtype=ms.float32), name='glb_GN')
        self.sub_GN = ms.Parameter(mint.zeros((1, 1, 1, self.image_dim_out * self.base_feat_height_reduction**2), dtype=ms.float32), name='sub_GN')

        dim_projection = hidden_size
        depth = 2
        layers = [
            mint.nn.Linear(image_dim_out * self.base_feat_height_reduction**2, dim_projection)
        ]
        for _ in range(1, depth):
            layers.extend([
                nn.GELU(),
                mint.nn.Linear(dim_projection, dim_projection)
            ])
        self.img_projection = nn.SequentialCell(layers)

        self.vocab_size = config.vocab_size
        self.img_features = None

        self.use_out_place_operations = False

    def get_img_features(self,
                         img_embeds: ms.Tensor,
                         attention_mask: Optional[ms.Tensor] = None) -> ms.Tensor:
        img_feature = self.img_processor(img_embeds, patch_attention_mask=attention_mask)

        if self.type_feature == "patch":
            patch_feature = img_feature

            use_token_compression = self.image_token_compression is not None
            use_padding = getattr(self, 'img_processor_padding',
                                  None) is not None
            if use_token_compression or use_padding:
                width = int(math.sqrt(patch_feature.shape[1]))
                patch_feature = patch_feature.view((-1, width, width, patch_feature.shape[-1]))
                patch_feature = mint.permute(patch_feature, (0, 3, 1, 2))

                if use_padding:
                    patch_feature = self.img_processor_padding(patch_feature)
                if use_token_compression:
                    patch_feature = self.image_token_compression(patch_feature)

                patch_feature = mint.permute(patch_feature, (0, 2, 3, 1))
                patch_feature = patch_feature.view(
                    (-1, patch_feature.shape[1] * patch_feature.shape[2], patch_feature.shape[-1])
                )
            return patch_feature

        raise NotImplementedError

    def construct(self, pixel_values: ms.Tensor,
                  image_sizes: ms.Tensor,
                  image_attention_mask: ms.Tensor) -> list[ms.Tensor]:
        img_sizes = image_sizes
        num_images, num_crops, c, h, w = pixel_values.shape
        bs = num_images
        pixel_values = mint.flatten(pixel_values, 0,1)

        img_features = self.get_img_features(
            pixel_values,
            mint.flatten(image_attention_mask.astype(ms.bool_), 0, 1)
        )

        base_feat_height_target = self.base_feat_height_target
        base_resolution = self.crop_size
        base_feat_height_reduction = self.base_feat_height_reduction

        base_feat_height = base_feat_width = int(np.sqrt(
            img_features.shape[1]))
        assert base_feat_height == base_feat_height_target \
            and base_feat_width == base_feat_height_target, \
                (f"base_feat_height: {base_feat_height}, "
                 f"base_feat_width: {base_feat_width}, "
                 f"expect {base_feat_height_target} features for hd transform")

        img_features = img_features.view((bs, -1,
                                         base_feat_height * base_feat_width,
                                         self.image_dim_out))
        C = self.image_dim_out
        H = base_feat_height

        output_imgs = []
        output_len = []
        if isinstance(img_sizes, ms.Tensor):
            img_sizes = img_sizes.view((-1, 2))
        for _bs in range(bs):
            h, w = int(img_sizes[_bs, 0].asnumpy()), int(img_sizes[_bs, 1].asnumpy())
            h = h // base_resolution
            w = w // base_resolution
            B_ = h * w

            global_img_feature = img_features[_bs, :1]

            glb_img = mint.reshape(global_img_feature, (1, H, H, C))
            glb_img = mint.reshape(glb_img, (1, H // base_feat_height_reduction, base_feat_height_reduction,
                                             H // base_feat_height_reduction, base_feat_height_reduction, C))
            glb_img = mint.permute(glb_img, (0, 1, 3, 2, 4, 5))
            glb_img = mint.reshape(glb_img, (1, H // base_feat_height_reduction,
                                             H // base_feat_height_reduction,
                                             base_feat_height_reduction * base_feat_height_reduction * C))
            temp_glb_GN = mint.tile(self.sub_GN, (1, H // base_feat_height_reduction, 1, 1))

            glb_img = mint.reshape(mint.concat([glb_img, temp_glb_GN], dim=2),
                                   (1, -1, base_feat_height_reduction * base_feat_height_reduction * C))

            sub_img = img_features[_bs, 1:]
            sub_img = sub_img[:B_]

            sub_img = mint.reshape(sub_img, (B_, H, H, C))
            sub_img = mint.reshape(sub_img, (B_, H // base_feat_height_reduction, base_feat_height_reduction,
                                             H // base_feat_height_reduction, base_feat_height_reduction, C))
            sub_img = mint.permute(sub_img, (0, 1, 3, 2, 4, 5))
            sub_img = mint.reshape(sub_img, (B_, -1, base_feat_height_reduction * base_feat_height_reduction * C))
            sub_img = mint.reshape(sub_img, (1, h, w, base_feat_height // base_feat_height_reduction,
                                             base_feat_width // base_feat_height_reduction, -1))
            sub_img = mint.permute(sub_img, (0, 1, 3, 2, 4, 5))
            sub_img = mint.reshape(sub_img, (1, h * base_feat_height // base_feat_height_reduction,
                                             w * base_feat_width // base_feat_height_reduction,
                                             base_feat_height_reduction * base_feat_height_reduction * C))

            if image_attention_mask is not None and image_attention_mask.shape[0] > 0:
                reshaped_image_attention_mask = image_attention_mask[
                    _bs, 1:B_ + 1, 0::2, 0::2]
                reshaped_image_attention_mask = mint.reshape(
                    reshaped_image_attention_mask,
                    (1, h, w, base_feat_height // base_feat_height_reduction,
                     base_feat_width // base_feat_height_reduction))
                reshaped_image_attention_mask = mint.permute(
                    reshaped_image_attention_mask, (0, 1, 3, 2, 4))
                reshaped_image_attention_mask = mint.reshape(
                    reshaped_image_attention_mask,
                    (1, h * base_feat_height // base_feat_height_reduction,
                     w * base_feat_width // base_feat_height_reduction))
                useful_height = int(mint.sum(reshaped_image_attention_mask[0, :, 0]).asnumpy())
                useful_width = int(mint.sum(reshaped_image_attention_mask[0, 0, :]).asnumpy())
                sub_img = sub_img[:, :useful_height, :useful_width]
                temp_sub_GN = mint.tile(self.sub_GN, (1, useful_height, 1, 1))
                temp_len = int(
                    mint.sum(image_attention_mask[_bs, :B_ + 1, 0::2, 0::2]).asnumpy()
                ) + (useful_height +
                     1) + base_feat_height // base_feat_height_reduction
            else:
                temp_sub_GN = mint.tile(
                    self.sub_GN, (1, h * base_feat_height // base_feat_height_reduction, 1, 1))
                temp_len = int((h * w + 1) * self.num_img_tokens + 1 +
                               (h + 1) * base_feat_height //
                               base_feat_height_reduction)

            sub_img = mint.reshape(mint.concat([sub_img, temp_sub_GN], dim=2),
                                   (1, -1, base_feat_height_reduction * base_feat_height_reduction * C))

            if self.hd_transform_order == 'glb_sub':
                output_imgs.append(
                    mint.concat([glb_img, self.glb_GN, sub_img], dim=1))
            elif self.hd_transform_order == 'sub_glb':
                output_imgs.append(
                    mint.concat([sub_img, self.glb_GN, glb_img], dim=1))
            else:
                raise NotImplementedError(
                    f'hd_transform_order = {self.hd_transform_order}, not implemented')

            assert temp_len == output_imgs[-1].shape[1], \
                f'temp_len: {temp_len}, output_imgs[-1].shape[1]: {output_imgs[-1].shape[1]}'

            output_len.append(temp_len)

        img_set_tensor = []
        for _output_img in output_imgs:
            img_feature_proj = self.img_projection(_output_img)
            img_set_tensor.append(mint.squeeze(img_feature_proj, 0))

        return img_set_tensor


class Phi4MMImagePixelInputs(TypedDict):
    """
    Dimensions:
        - bn: Batch size * number of images
        - p: Number of patches (1 + num_patches)
        - c: Number of channels (3)
        - h: Height of each image patch
        - w: Width of each image patch
        - nc: Number of crops
        - H_mask: Height of attention mask
        - W_mask: Width of attention mask
    """

    type: Literal["pixel_values"]

    data: Union[ms.Tensor, List[ms.Tensor]]
    """Shape: `(bn, p, 3, h, w)`, where `p` may vary per batch and image (dynamic dimension)."""

    image_sizes: ms.Tensor
    """Shape: `(bn, 2)` — [height, width] for each image."""

    num_img_tokens: List[int]
    """Length: `bn` — number of image tokens per image."""

    image_attention_mask: ms.Tensor
    """Shape: `(bn, nc, 32, 32)` — spatial attention mask (H_mask=32, W_mask=32)."""



class Phi4MMAudioFeatureInputs(TypedDict):
    """
    Dimensions:
        - bn: Batch size * number of audios
        - t: Time frames (M)
        - 80: Mel filterbank bins
    """
    type: Literal["audio_features"]
    data: Union[ms.Tensor, List[ms.Tensor]]
    """Shape: (bn, t, 80), where t may vary per batch (dynamic dimension)"""


class Phi4MMAudioEmbeddingInputs(TypedDict):
    """
    Dimensions:
        - b: Batch size
        - n: Number of audios
        - f: Audio feature size
        - h: Hidden size (must match language model backbone)
    """
    type: Literal["audio_embeds"]
    data: NestedTensors
    """Shape: (b, n, f, h)"""

Phi4MMAudioInputs = Union[Phi4MMAudioFeatureInputs, Phi4MMAudioEmbeddingInputs]

def cat_with_pad(tensors, dim, padding_value=0):
    """
    Concatenate tensors along `dim`, while pad to max for all other dims.
    """
    ndim = tensors[0].ndim
    assert all(t.ndim == ndim for t in tensors), "All tensors must have the same number of dimensions"

    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = mint.full(out_size, padding_value, dtype=tensors[0].dtype)

    index = 0
    for t in tensors:
        slices = []
        for d in range(ndim):
            if d == dim:
                slices.append(slice(index, index + t.shape[dim]))
            else:
                slices.append(slice(0, t.shape[d]))
        output[tuple(slices)] = t
        index += t.shape[dim]

    return output


class Phi4MMProcessingInfo(BaseProcessingInfo):

    @property
    def image_tokens(self) -> list[str]:
        return [f"<|image_{i+1}|>" for i in range(100)]

    @property
    def audio_tokens(self) -> list[str]:
        return [f"<|audio_{i+1}|>" for i in range(100)]
    
    def get_hf_processor(self, **kwargs: object) -> Phi4MMProcessor:
        return self.ctx.get_hf_processor(Phi4MMProcessor, **kwargs)

    def get_dynamic_hd(self, processor: Optional[ProcessorMixin] = None) -> int:
        if processor is None:
            processor = self.get_hf_processor()
        image_processor = processor.image_processor
        return image_processor.dynamic_hd

    def get_feature_extractor(self, **kwargs: object)-> SequenceFeatureExtractor:
        return self.get_hf_processor(**kwargs).audio_processor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None, "image": None}

    def _find_target_aspect_ratio(
        self,
        orig_width: int,
        orig_height: int,
        image_size: int,
        max_num: int,
        min_num: int,
    ):
        w_crop_num = math.ceil(orig_width / float(image_size))
        h_crop_num = math.ceil(orig_height / float(image_size))
        if w_crop_num * h_crop_num > max_num:
            aspect_ratio = orig_width / orig_height
            target_ratios = set((i, j) for i in range(1, max_num + 1)
                                for j in range(1, max_num + 1)
                                if i * j <= max_num and i * j >= min_num)
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
            image_processor = self.get_hf_processor().image_processor
            target_aspect_ratio = image_processor.find_closest_aspect_ratio(
                aspect_ratio,
                target_ratios,
                orig_width,
                orig_height,
                image_size,
            )
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
        else:
            target_width = image_size * w_crop_num
            target_height = image_size * h_crop_num
            target_aspect_ratio = (w_crop_num, h_crop_num)
        return target_aspect_ratio, target_height, target_width

    def _compute_num_image_tokens(
        self,
        orig_width: int,
        orig_height: int,
        dynamic_hd_size: int,
        vit_image_size: int,
        vit_patch_size: int,
        token_compression_factor: int = 2,
    ):
        assert vit_image_size % vit_patch_size == 0, (
            "vit_image_size must be divisible by vit_patch_size")
        assert (vit_image_size // vit_patch_size %
                token_compression_factor == 0), (
                    "vit_image_size // vit_patch_size must be divisible by "
                    "token_compression_factor")

        target_aspect_ratio, target_height, target_width = (
            self._find_target_aspect_ratio(orig_width,
                                           orig_height,
                                           vit_image_size,
                                           dynamic_hd_size,
                                           min_num=1))
        assert target_aspect_ratio[0] * vit_image_size == target_width, (
            f"{target_aspect_ratio[0]} * {vit_image_size} != {target_width}")
        assert target_aspect_ratio[1] * vit_image_size == target_height, (
            f"{target_aspect_ratio[1]} * {vit_image_size} != {target_height}")
        assert (target_height % vit_image_size == 0
                and target_width % vit_image_size == 0)

        padding_height, padding_width = _get_padding_size(
            orig_width, orig_height, target_height, target_width)
        assert padding_width == 0 or padding_height == 0, \
            "padding_width or padding_height must be 0"

        target_feat_width = target_width // vit_patch_size
        target_feat_height = target_height // vit_patch_size
        if padding_width >= vit_patch_size:
            assert padding_height == 0, "padding_height not 0"
            non_pad_feat_width = target_feat_width - math.floor(
                padding_width / vit_patch_size)
            non_pad_feat_height = target_feat_height
        elif padding_height >= vit_patch_size:
            assert padding_width == 0, "padding_width not 0"
            non_pad_feat_height = target_feat_height - math.floor(
                padding_height / vit_patch_size)
            non_pad_feat_width = target_feat_width
        else:
            non_pad_feat_width = target_feat_width
            non_pad_feat_height = target_feat_height

        feat_width = non_pad_feat_width // token_compression_factor
        feat_height = non_pad_feat_height // token_compression_factor
        if non_pad_feat_width % token_compression_factor != 0:
            feat_width += 1
        if non_pad_feat_height % token_compression_factor != 0:
            feat_height += 1
        num_hd_patch_tokens = feat_width * feat_height
        num_hd_newline_tokens = feat_height
        vit_feature_size = vit_image_size // vit_patch_size
        num_global_image_tokens = (vit_feature_size //
                                   token_compression_factor)**2
        num_sep_tokens = 1
        num_global_image_newline_tokens = \
            vit_feature_size // token_compression_factor

        return (num_global_image_tokens + num_sep_tokens +
                num_hd_patch_tokens + num_hd_newline_tokens +
                num_global_image_newline_tokens)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[ProcessorMixin] = None,
    ) -> int:
        hf_config = self.get_hf_config()
        vision_encoder_name = hf_config.img_processor
        if vision_encoder_name is None:
            vision_encoder_name = SIGLIP_NAME
        prepro_config = VISION_ENCODER_TO_PROCESSING_CONFIG[
            vision_encoder_name]
        vit_image_size = prepro_config['vit_image_size']
        vit_patch_size = prepro_config['vit_patch_size']
        token_compression_factor = prepro_config['token_compression_factor']

        dynamic_hd_size = self.get_dynamic_hd(processor=processor)

        image_num_tokens = self._compute_num_image_tokens(
            image_width,
            image_height,
            dynamic_hd_size=dynamic_hd_size,
            vit_image_size=vit_image_size,
            vit_patch_size=vit_patch_size,
            token_compression_factor=token_compression_factor,
        )

        return image_num_tokens

    def get_image_size_with_most_features(
        self,
        processor: Optional[ProcessorMixin] = None,
    ) -> ImageSize:
        hf_config = self.get_hf_config()
        vision_encoder_name = hf_config.img_processor
        if vision_encoder_name is None:
            vision_encoder_name = SIGLIP_NAME
        prepro_config = VISION_ENCODER_TO_PROCESSING_CONFIG[
            vision_encoder_name]
        vit_image_size = prepro_config['vit_image_size']

        max_side = vit_image_size * self.get_dynamic_hd(processor=processor)
        return ImageSize(height=max_side, width=vit_image_size)

    def get_audio_num_frames(self, audio_len: int, sr: float) -> int:
        if sr > 16000:
            audio_len //= sr // 16000
        elif 8000 <= sr < 16000:
            audio_len *= 2
        elif sr < 8000:
            raise RuntimeError(f"Unsupported sample rate {sr}")

        win_length = 400
        hop_length = 160

        num_frames = (audio_len - win_length) // hop_length + 1
        if num_frames < 1:
            raise ValueError("Waveform too short for given parameters.")

        return num_frames

    def _compute_audio_embed_size(self, audio_frames: int) -> int:
        hf_config = self.get_hf_config()
        compression_rate = hf_config.embd_layer['audio_embd_layer']['compression_rate']
        qformer_compression_rate = 1
        integer = audio_frames // compression_rate
        remainder = audio_frames % compression_rate

        result = integer if remainder == 0 else integer + 1

        integer = result // qformer_compression_rate
        remainder = result % qformer_compression_rate
        result = integer if remainder == 0 else integer + 1

        return result


class Phi4MMDummyInputsBuilder(BaseDummyInputsBuilder[Phi4MMProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        num_images = mm_counts.get("image", 0)
        image_tokens = self.info.image_tokens[:num_images]
        audio_tokens = self.info.audio_tokens[:num_audios]
        return "".join(image_tokens + audio_tokens)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        num_images = mm_counts.get("image", 0)

        target_width, target_height = self.info.get_image_size_with_most_features()

        mm_data = {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images
            ),
            "audio": self._get_dummy_audios(
                length=_AUDIO_MAX_SOUNDFILE_SIZE,
                num_audios=num_audios
            ),
        }
        return mm_data


class Phi4MMMultiModalProcessor(BaseMultiModalProcessor[Phi4MMProcessingInfo]):

    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return MultiModalDataParser(target_sr=feature_extractor.sampling_rate,
                                    audio_resample_method="scipy")

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    )-> BatchFeature:

        if not mm_data:
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=ms.Tensor([prompt_ids])))  # return BatchFeature(dict(input_ids=[prompt_ids])) # tensor_type="ms", tensor_type="np"

        sr = self.info.get_feature_extractor(**mm_kwargs).sampling_rate
        audio_data = mm_data.get("audios", [])
        if audio_data:
            mm_data['audios'] = [(data, sr) for data in audio_data]

        processed_outputs = super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

        num_img_tokens = [
            self.info.get_num_image_tokens(image_width=img_size[0],
                                           image_height=img_size[1])
            for img_size in processed_outputs["image_sizes"]
        ]
        processed_outputs["num_img_tokens"] = num_img_tokens

        audio_features = processed_outputs['input_audio_embeds']
        feature_sizes = [
            self.info.get_audio_num_frames(len(audio), sr)
            for audio in audio_data
        ]
        processed_outputs['input_audio_embeds'] = [
            audio_features[idx][:size]
            for idx, size in enumerate(feature_sizes)
        ]

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            input_image_embeds=MultiModalFieldConfig.batched("image"),
            image_attention_mask=MultiModalFieldConfig.batched("image"),
            image_sizes=MultiModalFieldConfig.batched("image"),
            num_img_tokens=MultiModalFieldConfig.batched("image"),
            input_audio_embeds=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptReplacement]:
        image_tokens: list[str] = self.info.image_tokens
        audio_tokens: list[str] = self.info.audio_tokens
        feature_extractor = self.info.get_feature_extractor(**hf_processor_mm_kwargs)
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        def get_image_replacement_phi4mm(item_idx: int):
            images = mm_items.get_items("image", (ImageEmbeddingItems, ImageProcessorItems))
            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    processor=hf_processor,
                )
            return [_IMAGE_PLACEHOLDER_TOKEN_ID] * num_image_tokens

        def get_audio_replacement_phi4mm(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            audio_len = audios.get_audio_length(item_idx)
            audio_frames = self.info.get_audio_num_frames(
                audio_len, feature_extractor.sampling_rate)
            audio_embed_size = self.info._compute_audio_embed_size(audio_frames)
            return [_AUDIO_PLACEHOLDER_TOKEN_ID] * audio_embed_size

        return [
            PromptReplacement(
                modality="image",
                target=image_tokens.__getitem__,
                replacement=get_image_replacement_phi4mm,
            ),
            PromptReplacement(
                modality="audio",
                target=audio_tokens.__getitem__,
                replacement=get_audio_replacement_phi4mm,
            ),
        ]

    def _recompute_cached_prompt_update(
        self,
        cached_update: ResolvedPromptUpdate,
        new_item_idx: int,
    ) -> ResolvedPromptUpdate:
        new_update = super()._recompute_cached_prompt_update(
            cached_update,
            new_item_idx,
        )

        if cached_update.modality == "image":
            image_tokens: list[str] = self.info.image_tokens
            new_update = new_update.with_target(image_tokens[new_item_idx])
        elif cached_update.modality == "audio":
            audio_tokens: list[str] = self.info.audio_tokens
            new_update = new_update.with_target(audio_tokens[new_item_idx])

        return new_update


@MULTIMODAL_REGISTRY.register_processor(
    Phi4MMMultiModalProcessor,
    info=Phi4MMProcessingInfo,
    dummy_inputs=Phi4MMDummyInputsBuilder,
)

class Phi4MMForCausalLM(NativeModel, SupportsLoRA, SupportsMultiModal):
    """
    Implements the Phi-4-multimodal-instruct model in vLLM.
    """
    packed_modules_mapping = {
        "qkv_proj": [
            "qkv_proj",
        ],
        "gate_up_proj": [
            "gate_up_proj",
        ],
    }


    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "base_layer.": "",
        },
        orig_to_new_prefix={
            "model.embed_tokens_extend.audio_embed.audio_projection.vision.":
            "embed_tokens_extend.audio_projection_for_vision.",
            "model.embed_tokens_extend.audio_embed.audio_projection.speech.":
            "embed_tokens_extend.audio_projection.",
            "model.embed_tokens_extend.audio_embed.": "embed_tokens_extend.",
            "model.embed_tokens_extend.image_embed.": "vision_encoder.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return f"<|image_{i}|>"
        if modality.startswith("audio"):
            return f"<|audio_{i}|>"

        raise ValueError("Only image or audio modality is supported")


    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config, "multimodal_config is required"
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config
        self.lora_config = lora_config

        
        assert get_pp_group().world_size == 1, "pipeline parallel is not supported"

        self.vision_encoder = Phi4MMImageEncoder(
            config,
            quant_config,
            prefix="model.vision_embed_tokens",
            model_dir=getattr(config, "_name_or_path", "")
        )

        if isinstance(config.embd_layer["audio_embd_layer"], dict):
            embedding_config = {
                "embedding_cls": config.embd_layer["audio_embd_layer"]["embedding_cls"],
                **config.embd_layer["audio_embd_layer"],
            }
        else:
            embedding_config = {
                "embedding_cls": config.embd_layer["embedding_cls"]
            }

        self.embed_tokens_extend = AudioEmbedding(config, **embedding_config)
        self.language_model = LlamaForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            ) 
        self.model = self.language_model.model

        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.common_preprocess(vllm_config, prefix)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)

    def common_preprocess(self, vllm_config, prefix=""):
        self.set_modules({
            "model": self.language_model.model,
            "vision_encoder": self.vision_encoder,
            "embed_tokens_extend": self.embed_tokens_extend,  
            "lm_head": self.lm_head
        })

        num_layers = getattr(self.config, "num_hidden_layers", 32)
        self.kv_caches = [
            AttentionWrapper()
            for _ in range(num_layers)
        ]

        
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(num_layers):
            compilation_config.static_forward_context[str(i)] = self.kv_caches[i]

    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> Optional[Phi4MMAudioInputs]:
        """
        Parse and validate the audio input to the model.  This handles both 
        audio features and audio embeddings, but only the former is used for
        now.

        Args:
            kwargs (object): Keyword arguments.

        Returns:
            Optional[Phi4MMAudioInputs]: Parsed and validated audio inputs.
        """
        audio_features = kwargs.pop("input_audio_embeds", None)
        audio_embeds = kwargs.pop("audio_embeds", None)

        if audio_features is None and audio_embeds is None:
            return None

        if audio_features is not None:
            return Phi4MMAudioFeatureInputs(type="audio_features",
                                            data=flatten_bn(audio_features))

        if audio_embeds is not None:
            return Phi4MMAudioEmbeddingInputs(type="audio_embeds",
                                              data=audio_embeds)

        raise AssertionError("This line should be unreachable.")
    
    def _process_audio_input(self, audio_input: Any, audio_projection_mode: str):
        """
        Create the audio embeddings from the audio input, where the audio input
        is pairs of audio features and audio embed lengths. The audio input is
        created by `input_mapper_for_phi4mm_audio`.

        Args:
            audio_input (Phi4MMAudioInputs): Audio input.

        Returns:
            NestedTensors: Audio embeddings
        """
        if audio_input["type"] == "audio_embeds":
            return audio_input["data"]

        audio_features = audio_input["data"]
        dtype = list(self.embed_tokens_extend.get_parameters())[0].dtype
        audio_embeds = [
            self.embed_tokens_extend(
                features.astype(dtype),
                audio_projection_mode=audio_projection_mode,
            ) for features in audio_features
        ]
        return audio_embeds

    def _parse_and_validate_image_input(self, **kwargs: object) -> Optional[Phi4MMImagePixelInputs]:
        input_image_embeds = kwargs.get("input_image_embeds")
        if input_image_embeds is None:
            return None

        image_sizes = kwargs.get("image_sizes")
        image_attention_mask = kwargs.get("image_attention_mask")
        num_img_tokens = kwargs.get("num_img_tokens")
        assert image_sizes is not None and image_attention_mask is not None and num_img_tokens is not None, "Missing image inputs"

        if is_list_of(input_image_embeds, ms.Tensor):
            assert all(len(p.shape) == 5 for p in input_image_embeds), "Incorrect image inputs"
            input_image_embeds = cat_with_pad(input_image_embeds, dim=0)
        elif isinstance(input_image_embeds, ms.Tensor):
            assert len(input_image_embeds.shape) == 6, "Incorrect image inputs"
            input_image_embeds = mint.flatten(input_image_embeds,0, 1)
        else:
            raise ValueError("Incorrect input_image_embeds inputs")

        if isinstance(image_attention_mask, list):
            image_attention_mask = cat_with_pad(image_attention_mask, dim=0)
        elif isinstance(image_attention_mask, ms.Tensor):
            image_attention_mask = mint.flatten(image_attention_mask,0, 1)
        else:
            raise ValueError("Incorrect image_attention_mask inputs")

        if isinstance(image_sizes, list):
            image_sizes = mint.cat(image_sizes, dim=0)
        elif isinstance(image_sizes, ms.Tensor):
            image_sizes = mint.flatten(image_sizes, 0, 1)
        else:
            raise ValueError("Incorrect image_sizes inputs")

        if isinstance(num_img_tokens, list):
            if all(isinstance(n, ms.Tensor) for n in num_img_tokens):
                num_img_tokens = [int(n.asnumpy()) for n in num_img_tokens]
            else:
                num_img_tokens = [int(n) for n in num_img_tokens]
        elif isinstance(num_img_tokens, ms.Tensor):
            num_img_tokens = mint.flatten(num_img_tokens, 0,1).tolist()
        else:
            raise ValueError("Incorrect num_img_tokens inputs")

        return Phi4MMImagePixelInputs(
            type="pixel_values",
            data=input_image_embeds,
            image_sizes=image_sizes,
            image_attention_mask=image_attention_mask,
            num_img_tokens=num_img_tokens,
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("input_image_embeds",
                            "image_embeds") and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(
                    **kwargs)
            if input_key in ("input_audio_embeds",
                            "audio_embeds") and "audios" not in modalities:
                modalities["audios"] = self._parse_and_validate_audio_input(
                    **kwargs)

        return modalities

    def _process_image_input(self, image_input: Phi4MMImagePixelInputs) -> List[ms.Tensor]:
        dtype = self.vision_encoder.parameters_dict()["img_projection.0.weight"].dtype
        pixel_values = image_input['data'].astype(dtype)
        np.save("debug_pixel_values.npy", pixel_values.cpu().numpy())
        image_sizes = image_input['image_sizes']
        image_attention_mask = image_input['image_attention_mask']
        image_embeds = self.vision_encoder(pixel_values, image_sizes, image_attention_mask)
        return image_embeds

    def get_multimodal_embeddings(self, **kwargs: object) -> MultiModalEmbeddings:  
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []

        multimodal_embeddings: tuple[ms.Tensor, ...] = ()

        audio_projection_mode = 'speech'
        for modality in modalities:
            if modality == "images":
                audio_projection_mode = "vision"
                image_input = modalities["images"]
                vision_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += tuple(vision_embeddings) 
            if modality == "audios":
                audio_input = modalities["audios"]
                audio_embeddings = self._process_audio_input(
                    audio_input, audio_projection_mode=audio_projection_mode)
                multimodal_embeddings += tuple(audio_embeddings)  

        return multimodal_embeddings 

    def get_input_embeddings(
        self,
        input_ids: ms.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> ms.Tensor:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            inputs_embeds = inputs_embeds.astype(ms.float32)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [_IMAGE_PLACEHOLDER_TOKEN_ID, _AUDIO_PLACEHOLDER_TOKEN_ID]
            )
            inputs_embeds = inputs_embeds.astype(ms.bfloat16)
        return inputs_embeds

    def get_input_embeddings_v0(
        self,
        input_ids: ms.Tensor,  
        image_input: Optional[Phi4MMImagePixelInputs] = None,
        audio_input: Optional[Phi4MMAudioFeatureInputs] = None,
    ) -> ms.Tensor:  
        audio_projection_mode = 'speech'
        inputs_embeds = self.get_input_embeddings(input_ids)
        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds,
                placeholder_token_id=_IMAGE_PLACEHOLDER_TOKEN_ID,  
            )
            audio_projection_mode = 'vision'

        if audio_input is not None:
            audio_embeds = self._process_audio_input(
                audio_input, audio_projection_mode=audio_projection_mode)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                audio_embeds,
                placeholder_token_id=_AUDIO_PLACEHOLDER_TOKEN_ID, 
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: ms.Tensor,  
        positions: ms.Tensor,  
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[ms.Tensor] = None, 
        **kwargs: object,
    ) -> ms.Tensor:  
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner from
        # `get_multimodal_embeddings` and `get_input_embeddings`, this
        # condition is only for v0 compatibility.
        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            audio_input = self._parse_and_validate_audio_input(**kwargs)

            
            if image_input is None and audio_input is None:
                inputs_embeds = None
            else:
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids,
                    image_input=image_input,
                    audio_input=audio_input)
                input_ids = None

        hidden_states, residual =  self.exec_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        return hidden_states

    def compute_logits(
        self,
        hidden_states: ms.Tensor,  
    ) -> Optional[ms.Tensor]:   
        logits = self.language_model.compute_logits(hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,ms.Tensor]]):
       
        params_dict = self.get_params_dict()
        if self.hf_to_vllm_mapper is not None:
            weights = self.hf_to_vllm_mapper.apply(weights)
        for name, weight in weights:
            if "model.embed_tokens_extend.image_embed" in name:
                self.vision_encoder.img_processor.load_weights([(name, weight)], params_dict)
            else:
                self.model.load_weights([(name, weight)], params_dict)
        return None
    
    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="model.",
            connector=["audio_projection_for_vision", "audio_projection"],
            tower_model=["vision_encoder", "embed_tokens_extend"],
        )

    def get_language_model(self) -> nn.Cell:
        return self.model
