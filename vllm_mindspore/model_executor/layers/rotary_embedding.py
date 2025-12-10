# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/llama/modeling_llama.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
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

import itertools
import math
from typing import Any, Optional, Union

import mindspore as ms
import numpy as np
from mindspore import Tensor, mint, nn, ops
from mindspore.common import dtype as mstype
from mindspore.ops.auto_generate.gen_ops_prim import SliceExt
from transformers import PretrainedConfig
from vllm.config import get_current_vllm_config

from vllm_mindspore.model_executor.models.vision import (
    get_llm_pos_ids_for_vision)
from vllm_mindspore.model_executor.utils import get_model_context
from vllm_mindspore.utils import MS_DTYPE_TO_SIZE, is_310p


def _get_feat_extract_output_lengths(input_lengths: ms.Tensor):
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = (((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 +
                      (input_lengths // 100) * 13)
    return feat_lengths, output_lengths


def _apply_rotary_emb(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    is_neox_style: bool,
) -> Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = mint.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return mint.cat((o1, o2), dim=-1)
    else:
        return mint.stack((o1, o2), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Cell):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        cache = cache.to(dtype)
        self.cos_sin_cache = cache

    def _compute_inv_freq(self, base: Union[int, float]) -> Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (base**(mint.arange(
            0, self.rotary_dim, 2, dtype=mstype.float32) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = mint.arange(self.max_position_embeddings, dtype=mstype.float32)

        freqs = ops.outer(t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = mint.cat((cos, sin), dim=-1)
        return cache

    def construct(
        self,
        positions: Tensor,
        query: Tensor,
        key: Tensor,
        batch_valid_length: Tensor,
        offsets: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """A PyTorch-native implementation of forward()."""
        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, axis=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = mint.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = mint.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key


class InferRotaryEmbedding(nn.Cell):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype,
    ) -> None:
        if not is_neox_style:
            raise NotImplementedError("InferRotaryEmbedding only support"
                                      "Neox-style rotary embeddings.")
        super().__init__()
        self.rotary_embedding_op = ops.ApplyRotaryPosEmb(2)
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.freqs_cos, self.freqs_sin = self._compute_cos_sin_cache()
        self.is_eager_mode = (
            get_current_vllm_config().model_config.enforce_eager)

    def _compute_inv_freq(self, base: Union[int, float]) -> Tensor:
        """
        Compute the inverse frequency with numpy.
        Numpy process is faster during initialization.
        """
        freqs_base = np.arange(0, self.rotary_dim,
                               2).astype(np.float32)  # (head_dim // 2, )
        freqs = 1.0 / (base**(freqs_base / self.rotary_dim)
                       )  # (head_dim // 2, )
        return freqs

    def _compute_cos_sin_cache(self) -> tuple[Tensor, Tensor]:
        freqs = self._compute_inv_freq(self.base)
        t = np.arange(0, self.max_position_embeddings, 1).astype(np.float32)
        freqs = np.outer(t, freqs)  # (max_position_embedding, head_dim // 2)
        emb = np.concatenate((freqs, freqs), axis=-1)
        freqs_cos = np.cos(emb)  # (seq_len, head_dim)
        freqs_sin = np.sin(emb)  # (seq_len, head_dim)
        freqs_cos = Tensor(freqs_cos, dtype=self.dtype)
        freqs_sin = Tensor(freqs_sin, dtype=self.dtype)
        return freqs_cos, freqs_sin

    def construct(
        self,
        positions: Tensor,
        query: Tensor,
        key: Tensor,
        batch_valid_length: Tensor,
        offsets: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if self.is_eager_mode:
            query = query.contiguous()
            key = key.contiguous()

        if get_model_context("is_prefill"):
            return self.rotary_embedding_op(query, key, self.freqs_cos,
                                            self.freqs_sin, batch_valid_length)

        freqs_cos = mint.index_select(self.freqs_cos, 0, positions)
        freqs_sin = mint.index_select(self.freqs_sin, 0, positions)
        return self.rotary_embedding_op(query, key, freqs_cos, freqs_sin,
                                        batch_valid_length)


class InferLlama3RotaryEmbedding(InferRotaryEmbedding):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype,
        scaling_factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        orig_max_position: int,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.orig_max_position = orig_max_position
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)

    def _compute_inv_freq(self, base: Union[int, float]) -> np.ndarray:
        inv_freqs = super()._compute_inv_freq(base)
        low_freq_wavelen = self.orig_max_position / self.low_freq_factor
        high_freq_wavelen = self.orig_max_position / self.high_freq_factor

        wave_len = 2 * math.pi / inv_freqs
        if self.low_freq_factor != self.high_freq_factor:
            smooth = (self.orig_max_position / wave_len - self.low_freq_factor
                      ) / (self.high_freq_factor - self.low_freq_factor)
        else:
            smooth = 0
        new_freqs = np.where(
            wave_len < high_freq_wavelen,
            inv_freqs,
            np.where(
                wave_len > low_freq_wavelen,
                inv_freqs / self.scaling_factor,
                (1 - smooth) * inv_freqs / self.scaling_factor +
                smooth * inv_freqs,
            ),
        )
        return new_freqs


class MRotaryEmbedding(RotaryEmbedding):
    """Rotary Embedding with Multimodal Sections."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: ms.Type,
        mrope_section: Optional[list[int]] = None,
        mrope_interleaved: bool = False,
    ) -> None:
        # In Qwen2.5-VL, the maximum index value is related to the duration of
        # the input video. We enlarge max_position_embeddings to 4 times to get
        # a larger the cos and sin cache.
        self.cache_max_position_num = max_position_embeddings * 4
        super().__init__(head_size, rotary_dim, self.cache_max_position_num,
                         base, is_neox_style, dtype)

        self.is_eager_mode = (
            get_current_vllm_config().model_config.enforce_eager)
        self.mrope_section = mrope_section
        if self.mrope_section:
            assert sum(self.mrope_section) == rotary_dim // 2

        self.mrope_interleaved = mrope_interleaved
        if self.mrope_interleaved:
            assert len(self.mrope_section) == 3
            mrope_section_np = np.array(self.mrope_section, dtype=np.int64)
            sec_total = mrope_section_np.sum()
            h_sec = np.array(list(range(1, self.mrope_section[1] * 3,
                                        3))) + sec_total
            w_sec = np.array(list(range(2, self.mrope_section[2] * 3,
                                        3))) + 2 * sec_total
            select_index = np.arange(sec_total, dtype=np.int64)
            select_index[1:mrope_section[1] * 3:3] = h_sec
            select_index[2:mrope_section[2] * 3:3] = w_sec
            self.rope_select_index = ms.from_numpy(select_index)
        else:
            assert len(self.mrope_section) == 3
            mrope_section_np = np.array(self.mrope_section, dtype=np.int64)
            sec_total = mrope_section_np.sum()
            sec_cu = mrope_section_np.cumsum()
            h_sec = np.arange(sec_cu[0], sec_cu[1]) + sec_total
            w_sec = np.arange(sec_cu[1], sec_cu[2]) + 2 * sec_total
            select_index = np.arange(sec_total, dtype=np.int64)
            select_index[sec_cu[0]:sec_cu[1]] = h_sec
            select_index[sec_cu[1]:sec_cu[2]] = w_sec
            self.rope_select_index = ms.from_numpy(select_index)

        #TODO(lvhaoyu): move import ms_custom_ops to the top of the file
        # after ms_custom_ops fix the issue that import ms_custom_ops
        # at the top of the file will cause aclrtSetDevice failure.
        try:
            import ms_custom_ops
            _ms_custom_ops_available = True
        except ImportError:
            _ms_custom_ops_available = False

        if self.is_neox_style and self.rotary_dim == self.head_size:
            # not interleave mode and not partial rope dim
            self.rotary_embedding_op = ops.ApplyRotaryPosEmb(2)
            self.rope_func = self._neox_style_and_full_rope
        elif is_310p() and _ms_custom_ops_available \
            and hasattr(ms_custom_ops, "apply_rotary_pos_emb_ms") \
            and not is_neox_style \
            and (((head_size - rotary_dim) *
                MS_DTYPE_TO_SIZE[dtype]) % 32 == 0) \
            and (((rotary_dim // 2) * MS_DTYPE_TO_SIZE[dtype]) % 32 == 0):
            # if on 310 device and custom_ops available and interleave mode
            self.rope_func = self._interleave_rope_310p
            from ms_custom_ops import apply_rotary_pos_emb_ms
            self.apply_rotary_pos_emb_ms = apply_rotary_pos_emb_ms
        else:
            self.rope_func = self._native_rope

    def apply_interleaved_rope(self, x: Tensor,
                               mrope_section: list[int]) -> Tensor:
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        """
        x = ops.transpose(x, (1, 0, 2))
        # mint.flatten ops in the ir is ShapeCalc + Reshape, and the
        # ShapeCalc is cpu ops, so we change it to shape and reshape for
        # aclgraph enable.
        # see https://gitee.com/mindspore/mindspore/issues/IDBWDX for details.
        t, _, _ = x.shape
        x = ops.reshape(x, (t, -1))
        x_t = mint.index_select(x, -1, self.rope_select_index)
        return x_t

    def apply_no_interleaved_rope(self, x: Tensor,
                                  mrope_section: list[int]) -> Tensor:
        """Apply non-interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        non-interleaved [TTTHHHWWW].
        """
        x = ops.transpose(x, (1, 0, 2))
        # mint.flatten ops in the ir is ShapeCalc + Reshape, and the
        # ShapeCalc is cpu ops, so we change it to shape and reshape for
        # aclgraph enable.
        # see https://gitee.com/mindspore/mindspore/issues/IDBWDX for details.
        t, _, _ = x.shape
        x = ops.reshape(x, (t, -1))
        x_t = mint.index_select(x, -1, self.rope_select_index)
        return x_t

    def _neox_style_and_full_rope(self, query, key, cos, sin, num_tokens,
                                  batch_valid_length):
        """For neox style rope and rotary_dim == head_size."""
        freqs_cos = mint.cat((cos, cos), dim=-1)
        freqs_sin = mint.cat((sin, sin), dim=-1)
        if self.is_eager_mode:
            query = query.contiguous()
            key = key.contiguous()
        query, key = self.rotary_embedding_op(query, key, freqs_cos, freqs_sin,
                                              batch_valid_length)
        return query, key

    def _native_rope(self, query, key, cos, sin, num_tokens,
                     batch_valid_length):
        """Naive implement, support all cases"""
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = mint.cat((query_rot, query_pass), dim=-1).view(query_shape)

        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = mint.cat((key_rot, key_pass), dim=-1).view(key_shape)
        return query, key

    def _interleave_rope_310p(self, query, key, cos, sin, num_tokens,
                              batch_valid_length):
        """For 310p device and interleave mode.
        Support rotary_dim < head_size"""
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        if self.is_eager_mode:
            query = query.contiguous()
            key = key.contiguous()
        #TODO(lvhaoyu):change to directly call
        # ms_custom_ops.apply_rotary_pos_emb_ms after ms_custom_ops fix the
        # issue that import ms_custom_ops at the top of the file will cause
        # aclrtSetDevice failure.
        # apply_rotary_pos_emb_ms is a inplace-op, so no need
        # to assign the output.
        self.apply_rotary_pos_emb_ms(query, key, cos, sin, "BSH", "interleave")
        query = query.view(query_shape)
        key = key.view(key_shape)
        return query, key

    def construct(
        self,
        positions: ms.Tensor,
        query: ms.Tensor,
        key: ms.Tensor,
        batch_valid_length: Tensor = None,
    ) -> tuple[ms.Tensor, ms.Tensor]:
        """
        Args:
            positions:
                [num_tokens,] (text only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        ######################################################################
        # max_pos: 128k, rotary_dim: 128
        # cos_sin_cache: (4*max_pos, rotary_dim//2 * 2)  # noqa: ERA001
        # positions: (3, 5120) # noqa: ERA001
        # cos_sin: (3, 5120, rotary_dim) # noqa: ERA001
        # cos/sin: cat[(1, 5120, mrope_sec),...] -> (1, 5120, rotary_dim//2)
        ######################################################################
        num_tokens = positions.shape[-1]
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = ops.chunk(cos_sin, 2, axis=-1)
        if positions.ndim == 2:
            if self.mrope_interleaved:
                cos = self.apply_interleaved_rope(cos, self.mrope_section)
                sin = self.apply_interleaved_rope(sin, self.mrope_section)
            else:
                cos = self.apply_no_interleaved_rope(cos, self.mrope_section)
                sin = self.apply_no_interleaved_rope(sin, self.mrope_section)

        query, key = self.rope_func(query, key, cos, sin, num_tokens,
                                    batch_valid_length)

        return query, key

    @staticmethod
    def get_input_positions(
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[list[list[int]], ms.Tensor],
        video_grid_thw: Union[list[list[int]], ms.Tensor],
        second_per_grid_ts: Optional[list[float]] = None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
        audio_feature_lengths: Optional[ms.Tensor] = None,
        use_audio_in_video: bool = False,
    ) -> tuple[list[list[int]], int]:
        """Get mrope input positions and delta value."""

        llm_positions, mrope_position_delta = \
            MRotaryEmbedding.get_input_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                context_len=context_len,
                seq_len=seq_len,
                audio_feature_lengths=audio_feature_lengths,
                use_audio_in_video=use_audio_in_video
            )

        return llm_positions.tolist(), mrope_position_delta

    @classmethod
    def get_input_positions_tensor(
        cls,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[list[list[int]], ms.Tensor],
        video_grid_thw: Union[list[list[int]], ms.Tensor],
        second_per_grid_ts: Optional[list[float]] = None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
        audio_feature_lengths: Optional[ms.Tensor] = None,
        use_audio_in_video: bool = False,
    ) -> tuple[ms.Tensor, int]:
        """Get mrope input positions and delta value."""
        from vllm.transformers_utils.config import thinker_uses_mrope
        if thinker_uses_mrope(hf_config):
            return cls._qwen3_omni_get_input_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                context_len=context_len,
                seq_len=seq_len,
                audio_feature_lengths=audio_feature_lengths,
                use_audio_in_video=use_audio_in_video,
            )
        elif hf_config.model_type in ["qwen3_vl", "qwen3_vl_moe"]:
            return cls._qwen3_vl_get_input_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                context_len=context_len,
                seq_len=seq_len,
            )
        elif hf_config.model_type in ["glm4v", "glm4v_moe"]:
            return cls._glm4v_get_input_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                context_len=context_len,
                seq_len=seq_len,
            )
        return cls._vl_get_input_positions_tensor(
            input_tokens=input_tokens,
            hf_config=hf_config,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            context_len=context_len,
            seq_len=seq_len,
        )

    @classmethod
    def _qwen3_vl_get_input_positions_tensor(
        cls,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: list[list[int]] | Tensor,
        video_grid_thw: list[list[int]] | Tensor,
        context_len: int = 0,
        seq_len: int | None = None,
        second_per_grid_ts: list[float] | None = None,
    ) -> tuple[Tensor, int]:
        """Get mrope input positions and delta value."""

        video_grid_thw = [[1, h, w] for t, h, w in video_grid_thw
                          for _ in range(t)]

        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size

        input_tokens_tensor = ms.tensor(input_tokens)
        vision_start_indices = ms.ops.argwhere(
            input_tokens_tensor == vision_start_token_id).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        llm_pos_ids_list: list = []

        st = 0
        remain_images, remain_videos = image_nums, video_nums

        image_index, video_index = 0, 0
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t,
                h // spatial_merge_size,
                w // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                mint.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            t_index = (mint.arange(llm_grid_t).view(-1, 1).expand(
                -1, llm_grid_h * llm_grid_w).flatten())
            h_index = (mint.arange(llm_grid_h).view(1, -1, 1).expand(
                llm_grid_t, -1, llm_grid_w).flatten())
            w_index = (mint.arange(llm_grid_w).view(1, 1, -1).expand(
                llm_grid_t, llm_grid_h, -1).flatten())
            llm_pos_ids_list.append(
                mint.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                mint.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = mint.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 -
                                len(input_tokens)).item()
        llm_positions = llm_positions[:, context_len:seq_len]
        return llm_positions, mrope_position_delta

    @classmethod
    def _glm4v_get_input_positions_tensor(
        cls,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[list[list[int]], Tensor],
        video_grid_thw: Union[list[list[int]], Tensor],
        context_len: int = 0,
        seq_len: Optional[int] = None,
    ) -> tuple[Tensor, int]:
        """Get mrope input positions and delta value for GLM4V (NumPy only)."""

        image_token_id = hf_config.image_token_id
        video_start_token_id = hf_config.video_start_token_id
        video_end_token_id = hf_config.video_end_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        llm_pos_ids_list: list[np.ndarray] = []

        if not (image_grid_thw is None and video_grid_thw is None):
            if isinstance(image_grid_thw, Tensor):
                image_grid_thw = image_grid_thw.asnumpy().tolist()

            input_token_type: list[str] = []
            video_check_flg = False
            for token in input_tokens:
                if token == video_start_token_id:
                    video_check_flg = True
                elif token == video_end_token_id:
                    video_check_flg = False

                if (token == image_token_id) and (video_check_flg is False):
                    input_token_type.append("image")
                elif (token == image_token_id) and (video_check_flg is True):
                    input_token_type.append("video")
                else:
                    input_token_type.append("text")

            input_type_group: list[tuple[str, int, int]] = []
            for key, group_iter in itertools.groupby(
                    enumerate(input_token_type), lambda x: x[1]):
                group_list = list(group_iter)
                start_index = group_list[0][0]
                end_index = group_list[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            video_frame_num = 1
            mm_data_idx = 0
            for modality_type, start_idx, end_idx in input_type_group:
                st_idx = int(llm_pos_ids_list[-1].max() +
                             1) if len(llm_pos_ids_list) > 0 else 0
                if modality_type == "image":
                    t, h, w = (
                        image_grid_thw[mm_data_idx][0],
                        image_grid_thw[mm_data_idx][1],
                        image_grid_thw[mm_data_idx][2],
                    )
                    llm_grid_t = int(t)
                    llm_grid_h = int(h // spatial_merge_size)
                    llm_grid_w = int(w // spatial_merge_size)

                    t_indices, h_indices, w_indices = np.meshgrid(
                        np.arange(llm_grid_t, dtype=np.int64),
                        np.arange(llm_grid_h, dtype=np.int64),
                        np.arange(llm_grid_w, dtype=np.int64),
                        indexing='ij')

                    stacked = np.stack([
                        t_indices.ravel(),
                        h_indices.ravel(),
                        w_indices.ravel()
                    ],
                                       axis=0) + st_idx

                    llm_pos_ids_list.append(stacked)
                    mm_data_idx += 1

                elif modality_type == "video":
                    t, h, w = (
                        video_frame_num,
                        image_grid_thw[mm_data_idx][1],
                        image_grid_thw[mm_data_idx][2],
                    )
                    llm_grid_t = int(t)
                    llm_grid_h = int(h // spatial_merge_size)
                    llm_grid_w = int(w // spatial_merge_size)

                    for t_idx in range(llm_grid_t):
                        t_indices, h_indices, w_indices = np.meshgrid(
                            np.arange(t_idx, dtype=np.int64),
                            np.arange(llm_grid_h, dtype=np.int64),
                            np.arange(llm_grid_w, dtype=np.int64),
                            indexing='ij')

                        stacked = np.stack([
                            t_indices.ravel(),
                            h_indices.ravel(),
                            w_indices.ravel()
                        ],
                                           axis=0) + st_idx

                        llm_pos_ids_list.append(stacked)

                    mm_data_idx += 1
                    video_frame_num += 1

                else:
                    text_len = int(end_idx - start_idx)
                    base = np.arange(text_len, dtype=np.int64)
                    stacked = np.tile(base, (3, 1)) + st_idx
                    llm_pos_ids_list.append(stacked)
                    video_frame_num = 1

        else:
            text_len = len(input_tokens)
            base = np.arange(text_len, dtype=np.int64)
            llm_pos_ids_list.append(np.tile(base, (3, 1)))

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        llm_positions = llm_positions[:, context_len:seq_len]
        mrope_position_delta = int(llm_positions.max() + 1 - len(input_tokens))
        return Tensor(llm_positions), mrope_position_delta

    @classmethod
    def _qwen3_omni_get_input_positions_tensor(
        cls,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: list[list[int]] | Tensor | None,
        video_grid_thw: list[list[int]] | Tensor | None,
        second_per_grid_ts: list[float] | None = None,
        context_len: int = 0,
        seq_len: int | None = None,
        audio_feature_lengths: Tensor | None = None,
        use_audio_in_video: bool = False,
    ) -> tuple[Tensor, int]:
        config = hf_config.thinker_config
        if isinstance(image_grid_thw, list):
            image_grid_thw = Tensor(image_grid_thw)
        if isinstance(video_grid_thw, list):
            video_grid_thw = Tensor(video_grid_thw)
        input_ids = Tensor(input_tokens)
        if input_ids is None or input_ids.ndim != 1:
            raise ValueError(
                "_omni3_get_input_positions_tensor expects 1D input_ids")

        seq_len = input_ids.shape[0]
        if audio_feature_lengths is not None and not isinstance(
                audio_feature_lengths, Tensor):
            audio_feature_lengths = Tensor(audio_feature_lengths,
                                           dtype=ms.int64)
        if second_per_grid_ts is None:
            if video_grid_thw is not None and video_grid_thw.numel() > 0:
                second_per_grids = mint.ones(video_grid_thw.shape[0],
                                             dtype=ms.float32)
            else:
                second_per_grids = Tensor([], dtype=ms.float32)
        else:
            second_per_grids = Tensor(second_per_grid_ts, dtype=ms.float32)

        spatial_merge_size = config.vision_config.spatial_merge_size
        image_token_id = config.image_token_id
        video_token_id = config.video_token_id
        audio_token_id = config.audio_token_id
        vision_start_token_id = config.vision_start_token_id
        audio_start_token_id = config.audio_start_token_id
        position_id_per_seconds = config.position_id_per_seconds

        vision_start_indices = ops.argwhere(
            input_ids == vision_start_token_id).squeeze(1)
        if vision_start_indices.numel() > 0:
            vision_tokens = input_ids[vision_start_indices + 1]
        else:
            vision_tokens = mint.empty((0, ), dtype=input_ids.dtype)
        audio_nums = mint.sum(input_ids == audio_start_token_id)
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = ((vision_tokens == audio_start_token_id).sum()
                      if use_audio_in_video else
                      (vision_tokens == video_token_id).sum())

        llm_pos_ids_list: list[Tensor] = []
        st = 0
        image_idx = 0
        video_idx = 0
        audio_idx = 0
        remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums  # noqa: E501
        multimodal_nums = (image_nums +
                           audio_nums if use_audio_in_video else image_nums +
                           video_nums + audio_nums)  # noqa: E501

        for _ in range(multimodal_nums):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            if (image_token_id in input_tokens or video_token_id
                    in input_tokens) and (remain_videos > 0
                                          or remain_images > 0):
                ed_vision_start = input_tokens.index(vision_start_token_id, st)
            else:
                ed_vision_start = len(input_tokens) + 1
            if audio_token_id in input_tokens and remain_audios > 0:
                ed_audio_start = input_tokens.index(audio_start_token_id, st)
            else:
                ed_audio_start = len(input_tokens) + 1
            min_ed = min(ed_vision_start, ed_audio_start)

            if min_ed == ed_audio_start:
                text_len = min_ed - st
                if text_len != 0:
                    st_idx = llm_pos_ids_list[-1].max(
                    ) + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        mint.arange(text_len, dtype=ms.int64).view(
                            1, -1).expand(3, -1) + st_idx)
                st_idx = llm_pos_ids_list[-1].max(
                ) + 1 if llm_pos_ids_list else 0
                bos_len = 1
                llm_pos_ids_list.append(
                    mint.arange(bos_len, dtype=ms.int64).view(1, -1).expand(
                        3, -1) + st_idx)
                st_idx = llm_pos_ids_list[-1].max(
                ) + 1 if llm_pos_ids_list else 0
                _, audio_len = _get_feat_extract_output_lengths(
                    audio_feature_lengths[audio_idx])
                llm_pos_ids = (mint.arange(audio_len, dtype=ms.int64).view(
                    1, -1).expand(3, -1) + st_idx)
                llm_pos_ids_list.append(llm_pos_ids)
                st_idx = llm_pos_ids_list[-1].max(
                ) + 1 if llm_pos_ids_list else 0
                eos_len = 1
                llm_pos_ids_list.append(
                    mint.arange(eos_len, dtype=ms.int64).view(1, -1).expand(
                        3, -1) + st_idx)
                st += text_len + bos_len + audio_len + eos_len
                audio_idx += 1
                remain_audios -= 1
            elif (min_ed == ed_vision_start
                  and input_ids[ed_vision_start + 1] == image_token_id):
                text_len = min_ed - st
                if text_len != 0:
                    st_idx = llm_pos_ids_list[-1].max(
                    ) + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        mint.arange(text_len, dtype=ms.int64).view(
                            1, -1).expand(3, -1) + st_idx)
                st_idx = llm_pos_ids_list[-1].max(
                ) + 1 if llm_pos_ids_list else 0
                bos_len = 1
                llm_pos_ids_list.append(
                    mint.arange(bos_len, dtype=ms.int64).view(1, -1).expand(
                        3, -1) + st_idx)
                st_idx = llm_pos_ids_list[-1].max(
                ) + 1 if llm_pos_ids_list else 0
                grid_t = image_grid_thw[image_idx][0]
                grid_hs = image_grid_thw[:, 1]
                grid_ws = image_grid_thw[:, 2]
                t_index = mint.arange(grid_t.item()) * position_id_per_seconds
                llm_pos_ids = get_llm_pos_ids_for_vision(
                    st_idx, image_idx, spatial_merge_size, t_index, grid_hs,
                    grid_ws)
                image_len = image_grid_thw[image_idx].prod() // (
                    spatial_merge_size**2)
                llm_pos_ids_list.append(llm_pos_ids)
                st_idx = llm_pos_ids_list[-1].max(
                ) + 1 if llm_pos_ids_list else 0
                eos_len = 1
                llm_pos_ids_list.append(
                    mint.arange(eos_len, dtype=ms.int64).view(1, -1).expand(
                        3, -1) + st_idx)
                st += text_len + bos_len + image_len + eos_len
                image_idx += 1
                remain_images -= 1
            elif (min_ed == ed_vision_start
                  and input_ids[ed_vision_start + 1] == video_token_id
                  and not use_audio_in_video):
                text_len = min_ed - st
                if text_len != 0:
                    st_idx = llm_pos_ids_list[-1].max(
                    ) + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        mint.arange(text_len, dtype=ms.int64).view(
                            1, -1).expand(3, -1) + st_idx)
                st_idx = llm_pos_ids_list[-1].max(
                ) + 1 if llm_pos_ids_list else 0
                bos_len = 1
                llm_pos_ids_list.append(
                    mint.arange(bos_len, dtype=ms.int64).view(1, -1).expand(
                        3, -1) + st_idx)
                st_idx = llm_pos_ids_list[-1].max(
                ) + 1 if llm_pos_ids_list else 0
                grid_t = video_grid_thw[video_idx][0]
                grid_hs = video_grid_thw[:, 1]
                grid_ws = video_grid_thw[:, 2]
                t_index = (mint.arange(grid_t.item()) *
                           float(second_per_grids[video_idx].item()) *
                           position_id_per_seconds)
                llm_pos_ids = get_llm_pos_ids_for_vision(
                    st_idx, video_idx, spatial_merge_size, t_index, grid_hs,
                    grid_ws)
                video_len = video_grid_thw[video_idx].prod() // (
                    spatial_merge_size**2)
                llm_pos_ids_list.append(llm_pos_ids)
                st_idx = llm_pos_ids_list[-1].max(
                ) + 1 if llm_pos_ids_list else 0
                eos_len = 1
                llm_pos_ids_list.append(
                    mint.arange(eos_len, dtype=ms.int64).view(1, -1).expand(
                        3, -1) + st_idx)
                st += text_len + bos_len + video_len + eos_len
                video_idx += 1
                remain_videos -= 1
            elif (min_ed == ed_vision_start
                  and ed_vision_start + 1 == ed_audio_start
                  and use_audio_in_video):
                text_len = min_ed - st
                if text_len != 0:
                    st_idx = llm_pos_ids_list[-1].max(
                    ) + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        mint.arange(text_len, dtype=ms.int64).view(
                            1, -1).expand(3, -1) + st_idx)
                st_idx = llm_pos_ids_list[-1].max(
                ) + 1 if llm_pos_ids_list else 0
                bos_len = 1
                bos_block = (mint.arange(bos_len, dtype=ms.int64).view(
                    1, -1).expand(3, -1) + st_idx)
                llm_pos_ids_list.append(bos_block)
                llm_pos_ids_list.append(bos_block)
                st_idx = llm_pos_ids_list[-1].max(
                ) + 1 if llm_pos_ids_list else 0
                _, audio_len = _get_feat_extract_output_lengths(
                    audio_feature_lengths[audio_idx])
                audio_llm_pos_ids = (mint.arange(
                    audio_len, dtype=ms.int64).view(1, -1).expand(3, -1) +
                                     st_idx)
                grid_t = video_grid_thw[video_idx][0]
                grid_hs = video_grid_thw[:, 1]
                grid_ws = video_grid_thw[:, 2]
                t_index = (mint.arange(grid_t.item()) *
                           float(second_per_grids[video_idx].item()) *
                           position_id_per_seconds)
                video_llm_pos_ids = get_llm_pos_ids_for_vision(
                    st_idx, video_idx, spatial_merge_size, t_index, grid_hs,
                    grid_ws)
                video_data_index, audio_data_index = 0, 0
                while (video_data_index < video_llm_pos_ids.shape[-1]
                       and audio_data_index < audio_llm_pos_ids.shape[-1]):
                    if (video_llm_pos_ids[0][video_data_index]
                            <= audio_llm_pos_ids[0][audio_data_index]):
                        llm_pos_ids_list.append(
                            video_llm_pos_ids[:, video_data_index:
                                              video_data_index + 1])
                        video_data_index += 1
                    else:
                        llm_pos_ids_list.append(
                            audio_llm_pos_ids[:, audio_data_index:
                                              audio_data_index + 1])
                        audio_data_index += 1
                if video_data_index < video_llm_pos_ids.shape[-1]:
                    llm_pos_ids_list.append(
                        video_llm_pos_ids[:,
                                          video_data_index:video_llm_pos_ids.
                                          shape[-1]])
                if audio_data_index < audio_llm_pos_ids.shape[-1]:
                    llm_pos_ids_list.append(
                        audio_llm_pos_ids[:,
                                          audio_data_index:audio_llm_pos_ids.
                                          shape[-1]])
                video_len = video_grid_thw[video_idx].prod() // (
                    spatial_merge_size**2)
                st_idx = llm_pos_ids_list[-1].max(
                ) + 1 if llm_pos_ids_list else 0
                eos_len = 1
                eos_block = (mint.arange(eos_len, dtype=ms.int64).view(
                    1, -1).expand(3, -1) + st_idx)
                llm_pos_ids_list.append(eos_block)
                llm_pos_ids_list.append(eos_block)
                st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2  # noqa: E501
                audio_idx += 1
                video_idx += 1
                remain_videos -= 1
                remain_audios -= 1

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                mint.arange(text_len.item(), dtype=ms.int64).view(
                    1, -1).expand(3, -1) + st_idx)

        llm_positions = mint.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        if llm_positions.shape[1] != seq_len:
            raise RuntimeError(
                "Position ids length mismatch with input ids length")

        mrope_position_delta = llm_positions.max() + 1 - seq_len
        return llm_positions, mrope_position_delta.item()

    @classmethod
    def _vl_get_input_positions_tensor(
        cls,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[list[list[int]], Tensor],
        video_grid_thw: Union[list[list[int]], Tensor],
        second_per_grid_ts: list[float],
        context_len: int = 0,
        seq_len: Optional[int] = None,
    ) -> tuple[Tensor, int]:
        """Get mrope input positions and delta value."""
        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        tokens_per_second = getattr(hf_config.vision_config,
                                    "tokens_per_second", 1.0)

        if isinstance(image_grid_thw, ms.Tensor):
            image_grid_thw = image_grid_thw.tolist()
        if isinstance(video_grid_thw, ms.Tensor):
            video_grid_thw = video_grid_thw.tolist()

        input_tokens_tensor = ms.Tensor(input_tokens)
        vision_start_indices = ops.argwhere(
            input_tokens_tensor == vision_start_token_id).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        llm_pos_ids_list: list = []

        st = 0
        remain_images, remain_videos = image_nums, video_nums

        image_index, video_index = 0, 0
        for _ in range(image_nums + video_nums):
            video_second_per_grid_t = 0.0
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_second_per_grid_t = 1.0
                if second_per_grid_ts is not None:
                    video_second_per_grid_t = second_per_grid_ts[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = \
                t, h // spatial_merge_size, w // spatial_merge_size
            text_len = ed - st

            llm_grid_t, llm_grid_h, llm_grid_w = \
                int(llm_grid_t), int(llm_grid_h), int(llm_grid_w)
            text_len = int(text_len)

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                mint.arange(0, text_len).view(1, -1).broadcast_to((3,
                                                                   -1)).int() +
                st_idx)

            t_index = (mint.arange(0, llm_grid_t).view(-1, 1).broadcast_to(
                (-1, llm_grid_h * llm_grid_w)) * video_second_per_grid_t *
                       tokens_per_second).int().flatten()
            h_index = mint.arange(0, llm_grid_h).view(1, -1, 1).broadcast_to(
                (llm_grid_t, -1, llm_grid_w)).flatten().int()
            w_index = mint.arange(0, llm_grid_w).view(1, 1, -1).broadcast_to(
                (llm_grid_t, llm_grid_h, -1)).flatten().int()

            llm_pos_ids_list.append(
                mint.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                mint.arange(0, text_len).view(1, -1).broadcast_to((3,
                                                                   -1)).int() +
                st_idx)

        llm_positions = mint.cat(llm_pos_ids_list, dim=1).view(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 -
                                len(input_tokens)).item()
        llm_positions = llm_positions[:, context_len:seq_len]

        return llm_positions, mrope_position_delta

    @staticmethod
    def get_next_input_positions(
        mrope_position_delta: int,
        context_len: int,
        seq_len: int,
    ) -> list[list[int]]:
        return [
            list(
                range(context_len + mrope_position_delta,
                      seq_len + mrope_position_delta)) for _ in range(3)
        ]

    @staticmethod
    def get_next_input_positions_tensor(out: ms.Tensor, out_offset: int,
                                        mrope_position_delta: int,
                                        context_len: int, num_new_tokens: int):
        values = mint.arange(
            int(mrope_position_delta + context_len),
            int(mrope_position_delta + context_len + num_new_tokens),
        ).broadcast_to((3, -1))
        out[:, out_offset:out_offset + num_new_tokens] = values


class InferMRotaryEmbedding(InferRotaryEmbedding):
    """Rotary Embedding with Multimodal Sections."""

    get_input_positions = MRotaryEmbedding.get_input_positions
    get_input_positions_tensor = MRotaryEmbedding.get_input_positions_tensor
    get_next_input_positions = MRotaryEmbedding.get_next_input_positions
    get_next_input_positions_tensor = \
        MRotaryEmbedding.get_next_input_positions_tensor

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: ms.Type,
        mrope_section: Optional[list[int]] = None,
    ) -> None:
        # In Qwen2.5-VL, the maximum index value is related to the duration of
        # the input video. We enlarge max_position_embeddings to 4 times to get
        # a larger the cos and sin cache.
        self.cache_max_position_num = max_position_embeddings * 4
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style  # type: ignore[assignment]
        self.dtype = dtype
        super().__init__(head_size, rotary_dim, self.cache_max_position_num,
                         base, is_neox_style, dtype)

        self.is_eager_mode = (
            get_current_vllm_config().model_config.enforce_eager)
        self.mrope_section = mrope_section
        if self.mrope_section:
            assert sum(self.mrope_section) == rotary_dim // 2

    def construct(  # type: ignore[override]
        self,
        positions: ms.Tensor,
        query: ms.Tensor,
        key: ms.Tensor,
        batch_valid_length: Tensor = None,
    ) -> tuple[ms.Tensor, ms.Tensor]:
        """
        Args:
            positions:
                [num_tokens,] (text only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        half_rotary_dim = self.rotary_dim // 2
        # prefill
        if get_model_context("is_prefill"):
            num_tokens = positions.shape[-1]
            cos, sin = self.freqs_cos[positions], self.freqs_sin[positions]
            cos = SliceExt()(cos, -1, 0, half_rotary_dim, 1)
            sin = SliceExt()(sin, -1, 0, half_rotary_dim, 1)
            if positions.ndim == 2:
                cos_l = mint.split(cos, self.mrope_section, dim=-1)
                sin_l = mint.split(sin, self.mrope_section, dim=-1)
                cos, sin = (), ()
                for i in range(len(
                        self.mrope_section)):  # type: ignore[arg-type]
                    cos_l_select = mint.index_select(cos_l[i], 0,
                                                     Tensor([i])).squeeze(0)
                    cos += (cos_l_select, )
                    sin_l_select = mint.index_select(sin_l[i], 0,
                                                     Tensor([i])).squeeze(0)
                    sin += (sin_l_select, )
                cos = mint.cat(cos, dim=-1)
                sin = mint.cat(sin, dim=-1)

            query_shape = query.shape
            query = query.view(num_tokens, -1, self.head_size)
            query_rot = SliceExt()(query, -1, 0, self.rotary_dim, 1)
            query_pass = SliceExt()(query, -1, self.rotary_dim,
                                    query_shape[-1], 1)
            query_rot = _apply_rotary_emb(query_rot, cos, sin,
                                          self.is_neox_style)
            query = mint.cat((query_rot, query_pass), dim=-1).view(query_shape)

            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_size)
            key_rot = SliceExt()(key, -1, 0, self.rotary_dim, 1)
            key_pass = SliceExt()(key, -1, self.rotary_dim, key_shape[-1], 1)
            key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
            key = mint.cat((key_rot, key_pass), dim=-1).view(key_shape)
            return query, key

        # decode
        if positions.ndim == 2:
            cos, sin = self.freqs_cos[positions], self.freqs_sin[positions]
            cos = SliceExt()(cos, -1, 0, half_rotary_dim, 1)
            sin = SliceExt()(sin, -1, 0, half_rotary_dim, 1)
            cos_l = mint.split(cos, self.mrope_section, dim=-1)
            sin_l = mint.split(sin, self.mrope_section, dim=-1)
            cos, sin = (), ()
            for i in range(len(self.mrope_section)):  # type: ignore[arg-type]
                cos_l_select = mint.index_select(cos_l[i], 0,
                                                 Tensor([i])).squeeze(0)
                cos += (cos_l_select, )
                sin_l_select = mint.index_select(sin_l[i], 0,
                                                 Tensor([i])).squeeze(0)
                sin += (sin_l_select, )
            cos = mint.cat(cos, dim=-1)
            sin = mint.cat(sin, dim=-1)
            freqs_cos = mint.cat([cos, cos], dim=-1).squeeze(1)
            freqs_sin = mint.cat([sin, sin], dim=-1).squeeze(1)
        else:
            positions = positions.flatten()
            freqs_cos = self.freqs_cos.index_select(0, positions)
            freqs_sin = self.freqs_sin.index_select(0, positions)
        if self.is_eager_mode:
            query = query.contiguous()
            key = key.contiguous()
        return self.rotary_embedding_op(query, key, freqs_cos, freqs_sin,
                                        batch_valid_length)


_ROPE_DICT: dict[tuple, Union[InferRotaryEmbedding, RotaryEmbedding]] = {}


def _yarn_get_mscale(scale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


def _yarn_find_correction_dim(num_rotations: int,
                              dim: int,
                              base: float = 10000,
                              max_position_embeddings: int = 2048) -> float:
    return (dim * math.log(max_position_embeddings /
                           (num_rotations * 2 * math.pi))) / (2 *
                                                              math.log(base))


# Find dim range bounds based on rotations
def _yarn_find_correction_range(
        low_rot: int,
        high_rot: int,
        dim: int,
        base: float = 10000,
        max_position_embeddings: int = 2048) -> tuple[int, int]:
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base,
                                  max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(low: float, high: float, dim: int,
                           dtype: np.dtype) -> np.ndarray:
    if low == high:
        high += 0.001  # Prevent singularity

    linear_func = (np.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = np.clip(linear_func, 0, 1)
    return ramp_func


class InferYaRNScalingRotaryEmbedding(InferRotaryEmbedding):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        scaling_factor: float,
        dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation
        self.mscale = float(
            _yarn_get_mscale(self.scaling_factor) * attn_factor)
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)

    def _compute_inv_freq(self, scaling_factor: float) -> Tensor:
        pos_freqs = self.base**(
            np.arange(0, self.rotary_dim, 2, dtype=np.float32) /
            self.rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow,
                                                self.rotary_dim, self.base,
                                                self.max_position_embeddings)
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (
            1 - _yarn_linear_ramp_mask(
                low,
                high,
                self.rotary_dim // 2,
                dtype=np.float32  # type: ignore[arg-type]
            )) * self.extrapolation_factor
        inv_freq = inv_freq_interpolation * (
            1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        return inv_freq

    def _compute_cos_sin_cache(self) -> tuple[Tensor, Tensor]:
        freqs = self._compute_inv_freq(self.scaling_factor)
        t = np.arange(self.max_position_embeddings *
                      self.scaling_factor).astype(np.float32)
        self.freqs = Tensor(freqs.reshape(1, 1, 1, -1), dtype=self.dtype)
        freqs = np.outer(t, freqs)  # (max_position_embedding, head_dim // 2)
        emb = np.concatenate((freqs, freqs), axis=-1)
        freqs_cos = np.cos(emb) * self.mscale  # (seq_len, head_dim)
        freqs_sin = np.sin(emb) * self.mscale  # (seq_len, head_dim)
        freqs_cos = Tensor(freqs_cos, dtype=self.dtype)
        freqs_sin = Tensor(freqs_sin, dtype=self.dtype)
        return freqs_cos, freqs_sin


class InferPhi3LongRoPEScaledRotaryEmbedding(nn.Cell):
    """Phi3 family of models scaled rotary embedding for MindSpore.

    Based on the original Phi3LongRoPEScaledRotaryEmbedding implementation.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        original_max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        short_factor: list[float],
        long_factor: list[float],
        dtype: mstype.Type,
        short_mscale: Optional[float] = None,
        long_mscale: Optional[float] = None,
    ):
        super().__init__()

        if is_neox_style is False:
            raise ValueError("`InferPhi3LongRoPEScaledRotaryEmbedding`"
                             " only supports neox_style.")

        self.rotary_dim = rotary_dim
        self.head_size = head_size
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        self.short_factor = short_factor
        self.long_factor = long_factor

        scale = self.max_position_embeddings / \
                self.original_max_position_embeddings
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = math.sqrt(
                1 + math.log(scale) /
                math.log(self.original_max_position_embeddings))
        if short_mscale is None:
            short_mscale = scaling_factor
        if long_mscale is None:
            long_mscale = scaling_factor

        self.short_mscale = short_mscale
        self.long_mscale = long_mscale

        short_cache = self._compute_cos_sin_cache(
            original_max_position_embeddings, short_factor, short_mscale)
        short_cache = short_cache.astype(dtype)

        long_cache = self._compute_cos_sin_cache(max_position_embeddings,
                                                 long_factor, long_mscale)
        long_cache = long_cache.astype(dtype)

        long_short_cache = ops.concat([short_cache, long_cache], axis=0)
        self.long_short_cos_sin_cache = ms.Tensor(long_short_cache)

    def _compute_inv_freq(self, rescale_factors: list[float]) -> Tensor:
        rescale_factors_array = np.array(rescale_factors, dtype=np.float32)
        rescale_factors_tensor = Tensor(rescale_factors_array)

        arange_result = ops.arange(0, self.rotary_dim, 2, dtype=mstype.float32)
        div_result = arange_result / self.rotary_dim
        power_result = ops.pow(self.base, div_result)
        inv_freq = 1.0 / (rescale_factors_tensor * power_result)
        return inv_freq

    def _compute_cos_sin_cache(
        self,
        max_position_embeddings: int,
        rescale_factors: list[float],
        mscale: float,
    ) -> Tensor:
        inv_freq = self._compute_inv_freq(rescale_factors)
        t = ops.arange(max_position_embeddings, dtype=mstype.float32)
        freqs = ops.outer(t, inv_freq)
        cos = ops.cos(freqs) * mscale
        sin = ops.sin(freqs) * mscale
        cache = ops.concat((cos, sin), axis=-1)
        return cache

    def _rotate_neox(self, x: Tensor) -> Tensor:
        """Rotates the input tensor for NeoX-style rotary embeddings.
        
        Args:
            x: Input tensor with shape [..., dim]
            
        Returns:
            Rotated tensor with the same shape
        """
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return ops.concat((-x2, x1), axis=-1)

    def construct(
        self,
        positions: Tensor,
        query: Tensor,
        key: Tensor,
        offsets: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        query = query.view(*query.shape[:-1], -1, self.head_size)
        key = key.view(*key.shape[:-1], -1, self.head_size)

        k = self.original_max_position_embeddings
        positions_greater_k = ops.gt(positions, k)
        any_positions_greater = ops.any(positions_greater_k)
        full_like_positions = ops.full_like(positions, k)
        long_prompt_offset = ops.cast(any_positions_greater,
                                      mstype.float32) * full_like_positions
        long_prompt_offset = ops.cast(long_prompt_offset, mstype.int64)

        idx = (positions + long_prompt_offset
               ) if long_prompt_offset is not None else positions
        idx = (idx + offsets) if offsets is not None else idx

        cos_sin = ops.gather(self.long_short_cos_sin_cache, idx, 0)

        cos, sin = ops.split(cos_sin, cos_sin.shape[-1] // 2, axis=-1)
        cos = ops.tile(cos, (1, 2)).expand_dims(-2)
        sin = ops.tile(sin, (1, 2)).expand_dims(-2)

        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        rotated_query = self._rotate_neox(query_rot)
        query_rot = query_rot * cos + rotated_query * sin
        query = ops.concat((query_rot, query_pass), axis=-1)

        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        rotated_key = self._rotate_neox(key_rot)
        key_rot = key_rot * cos + rotated_key * sin
        key = ops.concat((key_rot, key_pass), axis=-1)

        query_flat = query.view(*query.shape[:-2], -1)
        key_flat = key.view(*key.shape[:-2], -1)

        return query_flat, key_flat


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    is_neox_style: bool = True,
    rope_scaling: Optional[dict[str, Any]] = None,
    dtype: Optional[Any] = None,
    partial_rotary_factor: float = 1.0,
):
    if dtype is None:
        dtype = get_current_vllm_config().model_config.dtype

    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)

    key = (head_size, rotary_dim, max_position, base, is_neox_style,
           rope_scaling_args, dtype)
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]
    if rope_scaling is None:
        cls = InferRotaryEmbedding if is_neox_style else RotaryEmbedding
        rotary_emb = cls(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            dtype,
        )
    else:
        scaling_type = rope_scaling["rope_type"]

        if scaling_type == "llama3":
            scaling_factor = rope_scaling["factor"]
            low_freq_factor = rope_scaling["low_freq_factor"]
            high_freq_factor = rope_scaling["high_freq_factor"]
            original_max_position = rope_scaling[
                "original_max_position_embeddings"]
            rotary_emb = InferLlama3RotaryEmbedding(
                head_size, rotary_dim, max_position, base, is_neox_style,
                dtype, scaling_factor, low_freq_factor, high_freq_factor,
                original_max_position)
        elif scaling_type == "default":
            if "mrope_section" in rope_scaling:
                rotary_emb = MRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    dtype,
                    mrope_section=rope_scaling["mrope_section"],
                    mrope_interleaved=rope_scaling.get("mrope_interleaved",
                                                       False),
                )
            else:
                raise NotImplementedError
        elif scaling_type == "yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling[
                "original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("extrapolation_factor", "attn_factor", "beta_fast",
                         "beta_slow")
            }
            rotary_emb = InferYaRNScalingRotaryEmbedding(
                head_size, rotary_dim, original_max_position, base,
                is_neox_style, scaling_factor, dtype, **extra_kwargs)
        elif scaling_type == "longrope":
            short_factor = rope_scaling["short_factor"]
            long_factor = rope_scaling["long_factor"]
            original_max_position = rope_scaling[
                "original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("short_factor", "long_factor")
            }
            rotary_emb = InferPhi3LongRoPEScaledRotaryEmbedding(
                head_size, rotary_dim, max_position, original_max_position,
                base, is_neox_style, short_factor, long_factor, dtype)
        else:
            raise NotImplementedError

    _ROPE_DICT[key] = rotary_emb  # type: ignore[assignment]
    return rotary_emb
