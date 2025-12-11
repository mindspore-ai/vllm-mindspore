# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
"""Inference-only Qwen3VL model compatible with HuggingFace weights."""

import math
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import Any, Optional

import mindspore as ms
import mindspore.mint.nn.functional as F
import numpy as np
from mindspore import Parameter, Tensor, mint, nn, ops
from mindspore.ops.operations.nn_ops import (FlashAttentionScore,
                                             PromptFlashAttention)
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    smart_resize as image_smart_resize)
from transformers.models.qwen3_vl import (Qwen3VLProcessor,
                                          Qwen3VLVideoProcessor)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    smart_resize as video_smart_resize)
from transformers.video_utils import VideoMetadata
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.models import SupportsPP
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import (ImageSize, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        PromptReplacement, PromptUpdate,
                                        PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils import is_list_of

from vllm_mindspore.model_executor.layers.activation import (
    _ACTIVATION_REGISTRY)
from vllm_mindspore.model_executor.layers.linear import (ColumnParallelLinear,
                                                         RowParallelLinear)
from vllm_mindspore.model_executor.layers.logits_processor import (
    LogitsProcessor)
from vllm_mindspore.model_executor.layers.rotary_embedding import (
    _apply_rotary_emb)
from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead)
from vllm_mindspore.model_executor.model_loader.weight_utils import (
    default_weight_loader, get_loaded_weight)
from vllm_mindspore.model_executor.models.attention_mask import (
    MultiModalLowerTriangularMask)
from vllm_mindspore.model_executor.models.interfaces import (
    MultiModalEmbeddings, SupportsMultiModal)
from vllm_mindspore.model_executor.models.model_base import (AttentionWrapper,
                                                             NativeModel)
from vllm_mindspore.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionAttention, Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VLImageEmbeddingInputs, Qwen2_5_VLImageInputs,
    Qwen2_5_VLImagePixelInputs, Qwen2_5_VLVideoEmbeddingInputs,
    Qwen2_5_VLVideoInputs, Qwen2_5_VLVideoPixelInputs, Qwen2VLProcessingInfo)
from vllm_mindspore.model_executor.models.qwen3 import (Qwen3ForCausalLM,
                                                        Qwen3Model)
from vllm_mindspore.model_executor.models.utils import (
    PPMissingLayer, WeightsMapper, _merge_multimodal_embeddings, maybe_prefix)
from vllm_mindspore.utils import is_310p

try:
    from ms_custom_ops import apply_rotary_pos_emb_atb
    is_custom_rope_available = True
except ImportError:
    is_custom_rope_available = False

logger = init_logger(__name__)

# Official recommended max pixels is 24576 * 32 * 32
_MAX_FRAMES_PER_VIDEO = 24576


# extend Qwen2_5_VisionAttention to use custom ops for rope.
class Qwen3_VisionAttention(Qwen2_5_VisionAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_310p = is_310p()
        if self.is_310p:
            '''
            Use 'PromptFlashAttention' specifically for the 310p 
            to get higher precision.
            In 310p, 'pre_tokens' and 'next_tokens' only support '2147483647'
            and 'input_layout' only support 'BSH'.
            '''
            self.flash_attention_score = PromptFlashAttention(
                num_heads=self.num_attention_heads_per_partition,
                scale_value=1 / math.sqrt(self.hidden_size_per_attention_head),
                pre_tokens=2147483647,
                next_tokens=2147483647,
                input_layout="BSH",
                num_key_value_heads=self.num_attention_heads_per_partition,
                sparse_mode=0,
                inner_precise=0)
        else:
            self.flash_attention_score = FlashAttentionScore(
                head_num=self.num_attention_heads_per_partition,
                scale_value=1 / math.sqrt(self.hidden_size_per_attention_head),
                input_layout="TH")
        self.apply_rope = (self._custom_ops_rope if is_custom_rope_available
                           and not is_310p() else self._native_rope)

    def _native_rope(self, q, k, cos, sin, batch_valid_length):
        seq_length = q.shape[0]
        q = q.reshape(seq_length, self.num_attention_heads_per_partition,
                      self.hidden_size_per_attention_head)
        k = k.reshape(seq_length, self.num_attention_heads_per_partition,
                      self.hidden_size_per_attention_head)
        q = _apply_rotary_emb(q, cos, sin, True)
        k = _apply_rotary_emb(k, cos, sin, True)
        q = q.reshape(
            seq_length, self.num_attention_heads_per_partition *
            self.hidden_size_per_attention_head)
        k = k.reshape(
            seq_length, self.num_attention_heads_per_partition *
            self.hidden_size_per_attention_head)
        return q, k

    def _custom_ops_rope(self, q, k, cos, sin, batch_valid_length):
        cos = mint.cat((cos, cos), dim=-1)
        sin = mint.cat((sin, sin), dim=-1)
        q, k = apply_rotary_pos_emb_atb(q, k, cos, sin, batch_valid_length, 2,
                                        0)
        return q, k

    def construct(self, x: Tensor, batch_valid_length: Tensor,
                  position_embeddings: tuple[ms.Tensor, ms.Tensor],
                  q_seq_lens: Tensor) -> Tensor:
        qkv, _ = self.qkv(x)
        q, k, v = mint.split(
            qkv, (self.num_attention_heads_per_partition * self.head_dim,
                  self.num_attention_heads_per_partition * self.head_dim,
                  self.num_attention_heads_per_partition * self.head_dim), -1)
        cos, sin = position_embeddings
        origin_dtype = q.dtype

        q, k = self.apply_rope(q, k, cos, sin, batch_valid_length)

        # q/k reshape to TH
        q = q.astype(origin_dtype)
        k = k.astype(origin_dtype)
        if self.is_310p:
            batch_size = batch_valid_length.shape[0]
            max_seq_len = int(batch_valid_length.max())

            range_vector = mint.arange(max_seq_len,
                                       dtype=batch_valid_length.dtype)
            mask = range_vector.unsqueeze(0) < batch_valid_length.unsqueeze(1)

            indices = ops.nonzero(mask)
            target_shape = (batch_size, max_seq_len,
                            self.num_attention_heads_per_partition *
                            self.head_dim)

            q_padded = ops.scatter_nd(indices, q, target_shape)
            k_padded = ops.scatter_nd(indices, k, target_shape)
            v_padded = ops.scatter_nd(indices, v, target_shape)

            context_layer = self.flash_attention_score(q_padded, k_padded,
                                                       v_padded, None, None,
                                                       None, None, None, None,
                                                       None, None, None)
            context_layer = ops.gather_nd(context_layer, indices)
        else:
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
        self.dtype = get_current_vllm_config().model_config.dtype

        # Use Dense layer instead of Conv3d to accelerate the inference.
        self.proj = ms.nn.Dense(temporal_patch_size * patch_size * patch_size *
                                in_channels,
                                hidden_size,
                                has_bias=True,
                                dtype=self.dtype)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.proj(x)
        return x


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
        self.linear_fc1 = ColumnParallelLinear(in_features,
                                               hidden_features,
                                               bias=bias,
                                               quant_config=quant_config,
                                               return_bias=False,
                                               prefix=f"{prefix}.linear_fc1")
        self.linear_fc2 = RowParallelLinear(hidden_features,
                                            in_features,
                                            bias=bias,
                                            quant_config=quant_config,
                                            return_bias=False,
                                            prefix=f"{prefix}.linear_fc2")
        self.act_fn = act_fn

    def construct(self, x: ms.Tensor):
        mlp_output = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return mlp_output


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
        q_seq_lens: ms.Tensor,
    ) -> ms.Tensor:
        x = x + self.attn(self.norm1(x), batch_valid_length,
                          position_embeddings, q_seq_lens)

        x = x + self.mlp(self.norm2(x))
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
        self.norm = norm_layer(context_dim)
        self.linear_fc1 = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc1",
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = RowParallelLinear(
            self.hidden_size,
            d_model,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc2",
        )

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)

        x_parallel, _ = self.linear_fc1(x)
        x_parallel = self.act_fn(x_parallel)
        out, _ = self.linear_fc2(x_parallel)
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
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)

        self.patch_embed = Qwen3_VisionPatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )

        self.pos_embed = mint.nn.Embedding(self.num_position_embeddings,
                                           self.hidden_size,
                                           dtype=self.dtype)

        norm_layer = partial(mint.nn.LayerNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.CellList([
            Qwen3_VisionBlock(
                dim=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_dim=vision_config.intermediate_size,
                act_fn=_ACTIVATION_REGISTRY[vision_config.hidden_act],
                norm_layer=norm_layer,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{layer_idx}",
            ) for layer_idx in range(vision_config.depth)
        ])
        self.merger = Qwen3_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            norm_layer=norm_layer,
            spatial_merge_size=self.spatial_merge_size,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
        )
        if self.deepstack_visual_indexes is not None:
            self.deepstack_merger_list = nn.CellList([
                Qwen3_VisionPatchMerger(
                    d_model=vision_config.out_hidden_size,
                    context_dim=self.hidden_size,
                    spatial_merge_size=self.spatial_merge_size,
                    use_postshuffle_norm=True,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.deepstack_merger_list.{layer_idx}",
                ) for layer_idx in range(len(self.deepstack_visual_indexes))
            ])

    @property
    def dtype(self) -> ms.dtype:
        return self.patch_embed.proj.weight.dtype

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

        hidden_states = hidden_states + pos_embeds
        seq_len, _ = x.shape
        rotary_pos_emb = rotary_pos_emb.astype(hidden_states.dtype)
        emb = rotary_pos_emb
        position_embeddings = (mint.cos(emb), mint.sin(emb))

        hidden_states_list = []
        deepstack_visual_indexes = self.deepstack_visual_indexes

        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, batch_valid_length,
                                position_embeddings, q_seq_lens)
            if (deepstack_visual_indexes is not None
                    and layer_num in deepstack_visual_indexes):
                hidden_states_list.append(hidden_states)

        hidden_states = self.merger(hidden_states)

        # processing deepstack
        if deepstack_visual_indexes is not None:
            processed_hidden_states_list = [hidden_states]
            for idx, x in enumerate(hidden_states_list):
                x = self.deepstack_merger_list[idx](x)
                processed_hidden_states_list.append(x)
            # we cat the original visual features and deepstack features
            # along the feature dim
            hidden_states = mint.cat(
                processed_hidden_states_list,
                dim=1)  # [seq_len, hidden_size * (1 + depth_of_deepstack)]

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, Tensor]],
                     params_dict: dict[str, Parameter]) -> set[str]:
        stacked_params_mapping = [
            ("attn.qkv.", "attn.q.", "q"),
            ("attn.qkv.", "attn.k.", "k"),
            ("attn.qkv.", "attn.v.", "v"),
        ]
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                # Conv3d -> nn.Dense needs to flatten the weight
                if "patch_embed.proj.weight" in name:
                    loaded_weight = get_loaded_weight(loaded_weight)
                    loaded_weight = loaded_weight.reshape(
                        loaded_weight.shape[0], -1)
                    param.set_data(ms.Tensor(loaded_weight, dtype=param.dtype))
                else:
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def set_model_inputs(self):
        x_dtype = get_current_vllm_config().model_config.dtype
        dyn_x = ms.Tensor(shape=[None, None], dtype=x_dtype)
        dyn_batch_valid_length = ms.Tensor(shape=[None], dtype=ms.int32)
        dyn_q_seq_lens = ms.Tensor(shape=[None], dtype=ms.int32)
        dyn_rotary_pos_emb = ms.Tensor(shape=[None, None], dtype=ms.float32)
        dyn_pos_emb = ms.Tensor(shape=[None, None], dtype=x_dtype)

        self.set_inputs(dyn_x, dyn_batch_valid_length, dyn_q_seq_lens,
                        dyn_rotary_pos_emb, dyn_pos_emb)


class Qwen3VLProcessingInfo(Qwen2VLProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3VLConfig)

    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        return self.ctx.get_hf_processor(
            Qwen3VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )

    def get_tokenizer(self):
        return self.ctx.tokenizer

    def get_image_processor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).image_processor

    def get_video_processor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).video_processor

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 2,
        do_resize: bool = True,
        image_processor=None,
    ) -> tuple[ImageSize, int]:
        if image_processor is None and num_frames > 1:
            image_processor = self.get_video_processor()
        elif image_processor is None:
            image_processor = self.get_image_processor()

        is_video = isinstance(image_processor, Qwen3VLVideoProcessor)

        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        temporal_patch_size = vision_config.temporal_patch_size

        if do_resize:
            if is_video:
                smart_resize = video_smart_resize
                extra_kwargs = {
                    "num_frames": num_frames,
                    "temporal_factor": temporal_patch_size,
                }
            else:
                smart_resize = image_smart_resize
                extra_kwargs = {}
            resized_height, resized_width = smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * merge_size,
                min_pixels=image_processor.size["shortest_edge"],
                max_pixels=image_processor.size["longest_edge"],
                **extra_kwargs,
            )
            preprocessed_size = ImageSize(width=resized_width,
                                          height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width,
                                          height=image_height)

        padded_num_frames = num_frames + num_frames % temporal_patch_size

        grid_t = max(padded_num_frames // temporal_patch_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (merge_size**2)

        return preprocessed_size, num_vision_tokens

    def _get_max_video_frames(self,
                              max_tokens: int,
                              start_num_frames: int = 2) -> int:
        return super()._get_max_video_frames(max_tokens,
                                             start_num_frames=start_num_frames)

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        return super().get_num_frames_with_most_features(
            seq_len, mm_counts, max_frames_per_video=_MAX_FRAMES_PER_VIDEO)

    def get_max_video_tokens(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        target_width, target_height = self.get_image_size_with_most_features()
        video_soft_tokens = self.get_num_video_tokens(
            image_width=target_width,
            image_height=target_height,
            num_frames=self.get_num_frames_with_most_features(
                seq_len, mm_counts),
            image_processor=None,
        )

        # NOTE: By default in Qwen3-VL, one video token is converted to
        # "<{timestamp} seconds>" (on average 9.5 tokens) + vision_start_token + video_token + vision_end_token # noqa: E501
        formatted_video_soft_tokens = video_soft_tokens * 12.5
        return int(formatted_video_soft_tokens)

    def _calculate_timestamps(self, indices: list[int] | Tensor,
                              video_fps: float, merge_size: int):
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            # don't update metadata's frames_indices directly
            indices = indices + [indices[-1]
                                 ] * (merge_size - len(indices) % merge_size)
        timestamps = [idx / video_fps for idx in indices]
        timestamps = [(timestamps[i] + timestamps[i + merge_size - 1]) / 2
                      for i in range(0, len(timestamps), merge_size)]
        return timestamps

    def _get_video_second_idx(
        self,
        metadata: dict[str, Any],
        out_item: MultiModalKwargs,
        do_sample_frames: bool | None = None,
        sampled_fps: float | None = None,
    ) -> list[int]:
        video_processor = self.get_video_processor()
        merge_size = video_processor.merge_size
        indices = metadata["frames_indices"]

        # metadata["fps"] refers to the true fps of the input video.
        video_fps = metadata["fps"]
        if do_sample_frames is None:
            do_sample_frames = metadata.get("do_sample_frames", False)

        # If video frames are sampled in HF processor (instead of vLLM
        # video loader), we need to re-calculate the indices from original
        # metadata.
        if do_sample_frames:
            # here video_fps is the fps of the sampled video, and
            # metadata["fps"] refers to the fps of the original video.
            video_fps = sampled_fps if sampled_fps else video_processor.fps
            total_num_frames = metadata["total_num_frames"]
            num_frames = int(total_num_frames / metadata["fps"] * video_fps)
            num_frames = min(
                min(
                    max(num_frames, video_processor.min_frames),
                    video_processor.max_frames,
                ),
                total_num_frames,
            )
            indices = (np.linspace(0, total_num_frames - 1,
                                   num_frames).round().astype(int).tolist())
        timestamps = self._calculate_timestamps(indices, video_fps, merge_size)
        return timestamps


class Qwen3VLDummyInputsBuilder(BaseDummyInputsBuilder[Qwen3VLProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        video_token = "<|vision_start|><|video_pad|><|vision_end|>"

        return image_token * num_images + video_token * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        target_width, target_height = (
            self.info.get_image_size_with_most_features())
        target_num_frames = self.info.get_num_frames_with_most_features(
            seq_len, mm_counts)

        target_video_size, _ = self.info._get_vision_info(
            image_width=target_width,
            image_height=target_height,
            num_frames=target_num_frames,
            image_processor=self.info.get_video_processor(),
        )
        return {
            "image":
            self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
            ),
            "video":
            self._get_dummy_videos(
                width=target_video_size.width,
                height=target_video_size.height,
                num_frames=target_num_frames,
                num_videos=num_videos,
            ),
        }

    def _get_dummy_videos(
        self,
        *,
        width: int,
        height: int,
        num_frames: int,
        num_videos: int,
    ):
        num_frames = max(num_frames, 2)
        video = np.full((num_frames, width, height, 3), 255, dtype=np.uint8)
        video_items = []
        for i in range(num_videos):
            video_metadata = {
                "fps": 2.0,
                "duration": num_frames / 2.0,
                "total_num_frames": num_frames,
                "frames_indices": [i for i in range(num_frames)],
                "video_backend": "opencv",
                "do_sample_frames": False,
            }
            video_item = (video.copy(), video_metadata)
            video_items.append(video_item)
        return video_items


class Qwen3VLMultiModalProcessor(BaseMultiModalProcessor[Qwen3VLProcessingInfo]
                                 ):

    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(video_needs_metadata=True)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        processor = self.info.get_hf_processor(**mm_kwargs)

        # Separate video processing from image processing. Because the videos
        # are processed into serval image patches
        if ("videos" in mm_data and isinstance(mm_data["videos"], list)
                and len(mm_data["videos"]) > 0):
            video_grid_thw_lst = []
            pixel_values_videos_lst = []

            for item_idx, item in enumerate(mm_data.pop("videos", [])):
                video_array, metadata = item

                # NOTE: @JJJYmmm new attr metadata.frames_indices indicates
                # the sampled frames indices of pre-sampled videos, which is
                # used to calculate the timestamps. Make sure that
                # do_sample_frames in mm_kwargs is false for presampled videos.

                # NOTE: a copy of is created to update do_sample_frames,
                # otherwise mm_hash for the object will be incorrect.
                video_mm_kwargs = dict(**mm_kwargs)
                if "do_sample_frames" not in video_mm_kwargs:
                    # qwen_vl_utils already has "do_sample_frames" in
                    # mm_kwargs, don't overwrite it.
                    video_mm_kwargs["do_sample_frames"] = metadata.get(
                        "do_sample_frames", False)

                metadata = VideoMetadata(**{
                    k: metadata[k]
                    for k in metadata if k != "do_sample_frames"
                })

                video_mm_data = dict()
                video_mm_data["videos"] = [[video_array]]
                video_mm_data["video_metadata"] = [[metadata]]

                video_outputs = super()._call_hf_processor(
                    prompt="<|vision_start|><|video_pad|><|vision_end|>",
                    mm_data=video_mm_data,
                    mm_kwargs=video_mm_kwargs,
                    tok_kwargs=tok_kwargs)
                input_ids = video_outputs.pop("input_ids")
                video_placeholder = processor.tokenizer.batch_decode(
                    input_ids)[0]
                prompt = prompt.replace(
                    "<|vision_start|><|video_pad|><|vision_end|>",
                    video_placeholder,
                    1,
                )

                video_grid_thw_lst.append(video_outputs["video_grid_thw"])
                pixel_values_videos_lst.append(
                    video_outputs["pixel_values_videos"])
            video_outputs = dict(
                pixel_values_videos=mint.cat(pixel_values_videos_lst),
                video_grid_thw=mint.cat(video_grid_thw_lst),
            )
        else:
            video_outputs = dict()

        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        combined_outputs = dict(
            processed_outputs,
            **video_outputs,
        )
        return BatchFeature(combined_outputs)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_thw = hf_inputs.get("image_grid_thw", mint.empty((0, 3)))
        image_grid_sizes = image_grid_thw.prod(-1)

        video_grid_thw = hf_inputs.get("video_grid_thw", mint.empty((0, 3)))
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
        hf_config = self.info.get_hf_config()

        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        vision_end_token_id = hf_config.vision_end_token_id

        merge_length = image_processor.merge_size**2

        def get_image_replacement_qwen3vl(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data

            assert isinstance(grid_thw, Tensor)

            num_tokens = int(grid_thw.prod()) // merge_length
            return [hf_processor.image_token_id] * num_tokens

        def get_video_replacement_qwen3vl(item_idx: int):
            out_item = out_mm_kwargs["video"][item_idx]
            grid_thw = out_item["video_grid_thw"].data

            assert isinstance(grid_thw, Tensor)

            video, metadata = mm_items["video"][item_idx]
            do_sample_frames = hf_processor_mm_kwargs.get("do_sample_frames")
            sampled_fps = hf_processor_mm_kwargs.get("fps")
            if is_list_of(sampled_fps, float):
                sampled_fps = sampled_fps[item_idx]
            out_item = None
            timestamps = self.info._get_video_second_idx(
                metadata, out_item, do_sample_frames, sampled_fps)

            assert len(timestamps) == grid_thw[0], (
                f"The timestamps length({len(timestamps)}) should be equal "
                f"video length ({grid_thw[0]}).")

            frames_idx_token = [
                tokenizer.encode(f"<{curr_time:.1f} seconds>",
                                 add_special_tokens=False)
                for curr_time in timestamps
            ]
            num_tokens_per_frame = int(grid_thw[1:].prod()) // merge_length
            placeholder = []
            for frame_idx in frames_idx_token:
                placeholder.extend(frame_idx)
                placeholder.extend([vision_start_token_id] +
                                   [video_token_id] * num_tokens_per_frame +
                                   [vision_end_token_id])
            return PromptUpdateDetails.select_token_id(placeholder,
                                                       video_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=hf_processor.image_token,
                replacement=get_image_replacement_qwen3vl,
            ),
            # NOTE: We match string on purpose since searching sequence of
            # token ids takes more time.
            PromptReplacement(
                modality="video",
                target="<|vision_start|><|video_pad|><|vision_end|>",
                replacement=get_video_replacement_qwen3vl,
            ),
        ]


class Qwen3LLMModel(Qwen3Model):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 deepstack_layers: int = 0):
        # deepstack_layers is a param
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if not get_pp_group().is_first_rank:
            assert self.start_layer >= len(
                vllm_config.model_config.hf_config.vision_config.
                deepstack_visual_indexes), (
                    "start_layer should be greater than or equal to "
                    "len(deepstack_visual_indexes)")
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
            hidden_states, residual = layer(positions, hidden_states,
                                            key_caches[i - self.start_layer],
                                            value_caches[i - self.start_layer],
                                            slot_mapping, attn_mask,
                                            batch_valid_length, q_seq_lens,
                                            block_tables, residual)

            if deepstack_input_embeds is not None and i in range(
                    self.deepstack_layers):
                hidden_states = mint.add(hidden_states,
                                         deepstack_input_embeds[i])
        if get_pp_group().is_last_rank:
            hidden_states, residual = self.norm(hidden_states, residual)
        return hidden_states, residual


class Qwen3LLMForCausalLM(Qwen3ForCausalLM):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 deepstack_layers: int = 0):
        # 'deepstack_layers' is a parameters that indicates the number of
        # deepstack layers. The first 'deepstack_layers' layers will be add
        # with vit hidden states. Add this parameters to support the implement
        # in graph mode.
        super(Qwen3ForCausalLM, self).__init__(vllm_config=vllm_config)
        config = vllm_config.model_config.hf_config.text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = Qwen3LLMModel(vllm_config=vllm_config,
                                   prefix=prefix,
                                   deepstack_layers=deepstack_layers)
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size,
                                          quant_config=quant_config)
        else:
            self.lm_head = PPMissingLayer()
        if self.config.tie_word_embeddings and not is_310p():
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3VLProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3VLForConditionalGeneration(NativeModel, SupportsMultiModal,
                                      SupportsPP):
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

    #TODO: To support 'mm_encoder_tp_mode == "data"',
    # Linear in layers should be refactored first.
    supports_encoder_tp_data = False

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        })

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"

        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
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

        self.language_model = Qwen3LLMForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            deepstack_layers=self.deepstack_num_level)

        self.model = self.language_model.model
        self.lm_head = self.language_model.lm_head
        self.common_preprocess(vllm_config, prefix)

        # compile VocabEmbedding in language_model
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

    def common_preprocess(self, vllm_config, prefix=""):
        # override the common_preprocess method to
        # set modules prefix and casual mask.
        self.set_modules({
            "model.visual": self.visual,
            "model.language_model": self.language_model.model,
            "lm_head": self.language_model.lm_head
        })
        self.casual_mask = MultiModalLowerTriangularMask(
            dtype=self.model_config.dtype,
            max_model_len=self.model_config.max_model_len)
        self.kv_caches = [
            AttentionWrapper()
            for i in range(self.text_config.num_hidden_layers)
        ]

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(self.text_config.num_hidden_layers):
            compilation_config.static_forward_context[str(
                i)] = self.kv_caches[i]

    def _get_deepstack_input_embeds(self,
                                    num_tokens: int) -> IntermediateTensors:
        # get deepstack_input_embeds from buffer, and clear the buffer
        deepstack_input_embeds = \
            [self.deepstack_input_embeds[idx][:num_tokens]
             for idx in range(self.deepstack_num_level)]
        deepstack_input_embeds = mint.stack(deepstack_input_embeds, dim=0)
        return deepstack_input_embeds

    def _set_deepstack_input_embeds(self,
                                    deepstack_input_embeds: ms.Tensor) -> None:
        # set deepstack_input_embeds to buffer
        num_tokens = deepstack_input_embeds.shape[1]
        if num_tokens > self.deepstack_input_embeds[0].shape[0]:
            self.deepstack_input_embeds = [
                mint.zeros(
                    (num_tokens, self.config.text_config.hidden_size),
                    dtype=self.deepstack_input_embeds[0].dtype,
                ) for _ in range(self.deepstack_num_level)
            ]
        for idx in range(self.deepstack_num_level):
            self.deepstack_input_embeds[
                idx][:num_tokens] = deepstack_input_embeds[idx]

    def _clear_deepstack_input_embeds(self, num_tokens: int) -> None:
        # clear deepstack_input_embeds in buffer
        if num_tokens > 0:
            for idx in range(self.deepstack_num_level):
                self.deepstack_input_embeds[idx][:num_tokens].zero_()

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                        name: str) -> Tensor:
        if not isinstance(mm_input, (Tensor, list)):
            raise ValueError(
                f"Incorrect type of {name}. Got type: {type(mm_input)}")
        if isinstance(mm_input, Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(f"{name} should be 2D or batched 3D tensor. "
                                 f"Got ndim: {mm_input.ndim} "
                                 f"(shape={mm_input.shape})")
            return mm_input.reshape(-1, mm_input.shape[-1])
        else:
            return mint.concat(mm_input)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Qwen2_5_VLImageInputs | None:
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

            if not isinstance(pixel_values, (Tensor, list)):
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

            if not isinstance(image_embeds, Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            return Qwen2_5_VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

    def _parse_and_validate_video_input(
            self, **kwargs: object) -> Qwen2_5_VLVideoInputs | None:
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
                video_grid_thw=video_grid_thw,
            )

    def _process_image_input(self, image_input) -> tuple[Tensor, ...]:
        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            grid_thw = image_input["image_grid_thw"]
            assert grid_thw.ndim == 2

            rotary_pos_emb = self.rot_pos_emb(grid_thw)
            pos_emb = self.fast_pos_embed_interpolate(grid_thw.tolist())
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            grid_thw_1 = grid_thw.index_select(1, ms.Tensor([1])).reshape(-1)
            grid_thw_2 = grid_thw.index_select(1, ms.Tensor([2])).reshape(-1)
            grid_thw_0 = grid_thw.index_select(1, ms.Tensor([0])).reshape(-1)
            # compute batch_valid_length for custom ops
            batch_valid_length = mint.repeat_interleave(
                grid_thw_1 * grid_thw_2, grid_thw_0).astype(ms.int32)
            image_embeds = self.visual(pixel_values, batch_valid_length,
                                       batch_valid_length, rotary_pos_emb,
                                       pos_emb)
        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return image_embeds.split(sizes.tolist())

    def _process_video_input(
            self, video_input: Qwen2_5_VLVideoInputs) -> tuple[Tensor, ...]:
        if video_input["type"] == "video_embeds":
            return video_input["video_embeds"].type(self.visual.dtype)

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        pos_emb = self.fast_pos_embed_interpolate(grid_thw.tolist())
        pixel_values = video_input["pixel_values"].type(self.visual.dtype)
        grid_thw_1 = grid_thw.index_select(1, ms.Tensor([1])).reshape(-1)
        grid_thw_2 = grid_thw.index_select(1, ms.Tensor([2])).reshape(-1)
        grid_thw_0 = grid_thw.index_select(1, ms.Tensor([0])).reshape(-1)
        # compute batch_valid_length for custom ops
        batch_valid_length = mint.repeat_interleave(
            grid_thw_1 * grid_thw_2, grid_thw_0).astype(ms.int32)
        image_embeds = self.visual(pixel_values, batch_valid_length,
                                   batch_valid_length, rotary_pos_emb, pos_emb)
        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return image_embeds.split(sizes.tolist())

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}
        for input_key in kwargs:
            if (input_key in ("pixel_values", "image_embeds")
                    and "image" not in mm_input_by_modality):
                mm_input_by_modality[
                    "image"] = self._parse_and_validate_image_input(**kwargs)
            if (input_key in ("pixel_values_videos", "video_embeds")
                    and "video" not in mm_input_by_modality):
                mm_input_by_modality[
                    "video"] = self._parse_and_validate_video_input(**kwargs)
        return mm_input_by_modality

    def get_language_model(self):
        return self.language_model

    def get_multimodal_embeddings(
            self, **kwargs: object) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(
            **kwargs)
        if not mm_input_by_modality:
            return None

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[Tensor, ...] = ()

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

    def _compute_deepstack_embeds(
        self,
        inputs_embeds: ms.Tensor,
        multimodal_embeddings: MultiModalEmbeddings,
        is_multimodal: ms.Tensor,
    ) -> tuple[ms.Tensor, MultiModalEmbeddings]:
        visual_lens = [len(x) for x in multimodal_embeddings]
        multimodal_embeddings_cat = mint.cat(multimodal_embeddings, dim=0)

        (
            multimodal_embeddings_main,
            multimodal_embeddings_multiscale,
        ) = mint.split(
            multimodal_embeddings_cat,
            [self.visual_dim, self.multiscale_dim],
            dim=-1,
        )

        multimodal_embeddings = mint.split(multimodal_embeddings_main,
                                           visual_lens,
                                           dim=0)
        multimodal_embeddings_multiscale = mint.split(
            multimodal_embeddings_multiscale, visual_lens, dim=0)

        deepstack_input_embeds = inputs_embeds.new_zeros(
            inputs_embeds.shape[0],
            self.deepstack_num_level * inputs_embeds.shape[1])

        deepstack_input_embeds = _merge_multimodal_embeddings(
            inputs_embeds=deepstack_input_embeds,
            multimodal_embeddings=multimodal_embeddings_multiscale,
            is_multimodal=is_multimodal,
        )
        deepstack_input_embeds = deepstack_input_embeds.view(
            inputs_embeds.shape[0], self.deepstack_num_level, self.visual_dim)
        deepstack_input_embeds = deepstack_input_embeds.permute(1, 0, 2)

        return deepstack_input_embeds, multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: ms.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
    ) -> ms.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        placeholder_token_id = [
            self.config.image_token_id, self.config.video_token_id
        ]
        is_multimodal = ms.numpy.isin(input_ids, placeholder_token_id)

        if self.use_deepstack:
            (
                deepstack_input_embeds,
                multimodal_embeddings,
            ) = self._compute_deepstack_embeds(
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
        else:
            deepstack_input_embeds = None

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        if deepstack_input_embeds is not None:
            self._set_deepstack_input_embeds(deepstack_input_embeds)

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

        if (self.use_deepstack and inputs_embeds is not None
                and get_pp_group().is_first_rank):
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.shape[0])
        else:
            deepstack_input_embeds = None

        hidden_states, residual = self.exec_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            # args for deepstack
            deepstack_input_embeds=deepstack_input_embeds,
        )

        if inputs_embeds is not None and get_pp_group().is_first_rank:
            self._clear_deepstack_input_embeds(inputs_embeds.shape[0])

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        return hidden_states

    def compute_logits(
        self,
        hidden_states: ms.Tensor,
    ) -> ms.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   ms.Tensor]]) -> set[str]:
        params_dict = self.get_params_dict()
        loaded_param = set()
        visual_load = set()
        text_load = set()
        for name, weight in weights:
            if "model.visual." in name:
                visual_load.update(
                    self.visual.load_weights([(name, weight)], params_dict))
            elif "model.language_model." in name:
                text_load.update(
                    self.model.load_weights([(name, weight)], params_dict))
            else:
                # Handle other weights
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, weight)
                    loaded_param.add(name)
        loaded_param.update(visual_load)
        loaded_param.update(text_load)
        if self.language_model.config.tie_word_embeddings and not is_310p():
            loaded_param.add("lm_head.weight")
        return loaded_param

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="model.visual.merger",
            tower_model="model.visual.",
        )

    def _get_rot_pos_ids(self, h: int, w: int,
                         spatial_merge_size: int) -> ms.Tensor:
        """Generate rotary position IDs for vision patches with spatial merging.

        This function creates position indices (h_pos, w_pos) for each spatial
        location after applying spatial merging. The positions are reorganized
        to match the spatial merge pattern used in vision transformers.
        Results are cached for repeated (h, w) combinations to avoid redundant
        computation.

        Args:
            h (int): Height of the image grid in patches.
            w (int): Width of the image grid in patches.
            spatial_merge_size (int): Size of spatial merge window (e.g., 2
            means 2x2 patches are merged into one).

        Returns:
            ms.Tensor: Position IDs tensor of shape [h*w, 2], where each row
            contains [h_pos, w_pos] for the corresponding spatial location 
            after merge.

        Notes:
            - Cache hit rate is typically high in batch inference with uniform
              image sizes
            - The reshape/permute operations align positions with the merge
              pattern
            - Output positions are in int32 for efficient indexing
        """
        key = (h, w)
        # LRU cache: move to end if exists (most recently used)
        if key in self._rot_pos_ids_cache:
            cached = self._rot_pos_ids_cache[key]
            # Move to end (mark as most recently used)
            self._rot_pos_ids_cache.move_to_end(key)
            return cached

        hpos_ids = mint.arange(h, dtype=ms.int32).view(h, 1).expand(h, w)
        wpos_ids = mint.arange(w, dtype=ms.int32).view(1, w).expand(h, w)

        hpos_ids = (hpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        ).permute(0, 2, 1, 3).reshape(-1))
        wpos_ids = (wpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        ).permute(0, 2, 1, 3).reshape(-1))

        pos_ids = mint.stack([hpos_ids, wpos_ids], dim=-1).astype(ms.int32)
        # Add to cache (at end, most recently used)
        self._rot_pos_ids_cache[key] = pos_ids
        # Remove oldest entry if cache exceeds max size
        if len(self._rot_pos_ids_cache) > self._rot_pos_ids_cache_max_size:
            self._rot_pos_ids_cache.popitem(last=False)
        return pos_ids

    def rot_pos_emb(self, grid_thw: ms.Tensor) -> ms.Tensor:
        """Compute rotary position embeddings for vision inputs.

        This function generates rotary position embeddings for vision patches 
        in a multimodal model. It handles both images (t=1) and videos (t>1) by
        computing 2D spatial position embeddings and optionally repeating them
        across temporal dimension. The function is moved out of vision
        transformer to support graph compilation mode.

        Performance optimizations:
            1. Caches position IDs for repeated (h, w) combinations via
              _get_rot_pos_ids
            2. Avoids unnecessary tile operations when t=1 (single frame images)
            3. Computes rotary embeddings only once for max_grid_size
            4. Uses single host-to-device transfer via asnumpy().tolist()

        Args:
            grid_thw (ms.Tensor): Tensor of shape [N, 3] containing (t, h, w)
                for each image/video in the batch, where:
                - t: temporal dimension (number of frames)
                - h: height of the image grid in patches
                - w: width of the image grid in patches

        Returns:
            ms.Tensor: Rotary position embeddings of shape 
                [total_patches, embed_dim], where total_patches = 
                sum(t*h*w for each item in grid_thw) and
                embed_dim is the rotary embedding dimension (head_dim).

        Example:
            >>> # For 2 images: first is 224x224, second is 448x448
            >>> grid_thw = ms.Tensor([[1, 16, 16], [1, 32, 32]])
            >>> rot_emb = model.rot_pos_emb(grid_thw)
            >>> # Output shape: [(16*16) + (32*32), embed_dim]
        """
        # move out of vision transformer because it is not suit for graph mode.
        spatial_merge_size = self.vision_config.spatial_merge_size
        grid_list = grid_thw.asnumpy().tolist()
        pos_ids = []
        max_grid_size = 0

        for t, h, w in grid_list:
            max_grid_size = max(max_grid_size, h, w)
            base_ids = self._get_rot_pos_ids(h, w, spatial_merge_size)
            if t == 1:
                pos_ids.append(base_ids)
            else:
                pos_ids.append(mint.tile(base_ids, (t, 1)))

        pos_ids = mint.cat(pos_ids, dim=0)

        rotary_pos_emb_full = self.rotary_pos_emb_full(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def fast_pos_embed_interpolate(self,
                                   grid_thw: list[list[int]]) -> ms.Tensor:
        # move out of vision transformer because it is not suit for graph mode.
        num_grid_per_side = self.visual.num_grid_per_side
        m_size = self.visual.spatial_merge_size
        hidden_dim = self.visual.pos_embed.embedding_dim

        outputs = []
        for t, h, w in grid_thw:
            h_idxs = mint.linspace(0,
                                   num_grid_per_side - 1,
                                   h,
                                   dtype=ms.float32)
            w_idxs = mint.linspace(0,
                                   num_grid_per_side - 1,
                                   w,
                                   dtype=ms.float32)

            h_floor = h_idxs.astype(ms.int64)
            w_floor = w_idxs.astype(ms.int64)
            h_ceil = mint.clamp(h_floor + 1, 0, num_grid_per_side - 1)
            w_ceil = mint.clamp(w_floor + 1, 0, num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            # Create meshgrid view for all h, w vars
            dh_grid, dw_grid = mint.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = mint.meshgrid(h_floor,
                                                       w_floor,
                                                       indexing="ij")
            h_ceil_grid, w_ceil_grid = mint.meshgrid(h_ceil,
                                                     w_ceil,
                                                     indexing="ij")
            h_floor_grid_idx = h_floor_grid * num_grid_per_side
            h_ceil_grid_idx = h_ceil_grid * num_grid_per_side

            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - dw_grid + w11

            idx00 = h_floor_grid_idx + w_floor_grid
            idx01 = h_floor_grid_idx + w_ceil_grid
            idx10 = h_ceil_grid_idx + w_floor_grid
            idx11 = h_ceil_grid_idx + w_ceil_grid

            indices = mint.stack([idx00, idx01, idx10, idx11],
                                 dim=0).reshape(4, -1)
            weights = mint.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
            weights = weights.astype(self.visual.dtype)

            embeds = self.visual.pos_embed(indices)
            weighted_embeds = embeds * weights
            p0, p1, p2, p3 = weighted_embeds.unbind(dim=0)
            combined = p0 + p1 + p2 + p3

            combined = combined.view(h * w, hidden_dim)
            repeated = combined.unsqueeze(0).expand(t, -1, -1)
            repeated = repeated.view(t, h // m_size, m_size, w // m_size,
                                     m_size, hidden_dim)
            repeated = repeated.permute(0, 1, 3, 2, 4,
                                        5).reshape(-1, hidden_dim)
            outputs.append(repeated)

        return mint.cat(outputs, dim=0)

    def _get_extra_input_params(self):
        """Override to provide deepstack input embeds for multimodal models.
        
        Returns a list containing dyn_deepstack_input_embeds tensor.
        """
        if get_pp_group().is_first_rank:
            dyn_deepstack_input_embeds = Tensor(shape=[None, None, None],
                                                dtype=self.model_config.dtype)
        else:
            dyn_deepstack_input_embeds = None
        return [dyn_deepstack_input_embeds]
