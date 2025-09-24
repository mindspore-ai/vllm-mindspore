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

import mindspore
import numpy as np
from mindspore import Tensor, mint, nn, ops, tensor
from mindspore.common import dtype as mstype
from mindspore.ops.auto_generate.gen_ops_prim import SliceExt
from transformers import PretrainedConfig
from vllm.config import get_current_vllm_config

from vllm_mindspore.model_executor.utils import get_model_context
try:
    import ms_custom_ops
    ms_custom_ops_avail = True
except ImportError:
    ms_custom_ops_avail = False

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
        dtype: mindspore.Type,
        mrope_section: Optional[list[int]] = None,
    ) -> None:
        # In Qwen2.5-VL, the maximum index value is related to the duration of
        # the input video. We enlarge max_position_embeddings to 4 times to get
        # a larger the cos and sin cache.
        self.cache_max_position_num = max_position_embeddings * 4
        super().__init__(head_size, rotary_dim, self.cache_max_position_num,
                         base, is_neox_style, dtype)

        self.mrope_section = mrope_section
        if self.mrope_section:
            assert sum(self.mrope_section) == rotary_dim // 2

    def construct(
        self,
        positions: mindspore.Tensor,
        query: mindspore.Tensor,
        key: mindspore.Tensor,
        batch_valid_length: Tensor = None,
    ) -> tuple[mindspore.Tensor, mindspore.Tensor]:
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
            cos_l = mint.split(cos, self.mrope_section, dim=-1)
            sin_l = mint.split(sin, self.mrope_section, dim=-1)
            cos, sin = (), ()
            for i in range(len(self.mrope_section)):  # type: ignore[arg-type]
                cos += (cos_l[i][i], )
                sin += (sin_l[i][i], )
            cos = mint.cat(cos, dim=-1)
            sin = mint.cat(sin, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        if ms_custom_ops_avail is False:
            query_rot = query[..., :self.rotary_dim]
            query_pass = query[..., self.rotary_dim:]
            query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
            query = mint.cat((query_rot, query_pass), dim=-1).view(query_shape)

            key_rot = key[..., :self.rotary_dim]
            key_pass = key[..., self.rotary_dim:]
            key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
            key = mint.cat((key_rot, key_pass), dim=-1).view(key_shape)
        else:
            query, key = ms_custom_ops.apply_rotary_pos_emb_v3(query, key, cos, sin, "BSH", "interleave")
            query = query.view(query_shape)
            key = key.view(key_shape)
        return query, key

    @staticmethod
    def get_input_positions(
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[list[list[int]], mindspore.Tensor],
        video_grid_thw: Union[list[list[int]], mindspore.Tensor],
        second_per_grid_ts: Optional[list[float]] = None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
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
            )

        return llm_positions.tolist(), mrope_position_delta

    @classmethod
    def get_input_positions_tensor(
        cls,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[list[list[int]], mindspore.Tensor],
        video_grid_thw: Union[list[list[int]], mindspore.Tensor],
        second_per_grid_ts: Optional[list[float]] = None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
    ) -> tuple[mindspore.Tensor, int]:
        """Get mrope input positions and delta value."""
        if "glm4v" in hf_config.model_type:
            return cls._glm4v_get_input_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                context_len=context_len,
                seq_len=seq_len,
            )
        else:
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
    def _glm4v_get_input_positions_tensor(
        cls,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[list[list[int]], Tensor],
        video_grid_thw: Union[list[list[int]], Tensor],
        context_len: int = 0,
        seq_len: Optional[int] = None,
    ) -> tuple[Tensor, int]:
        """Get mrope input positions and delta value for GLM4V."""

        image_token_id = hf_config.image_token_id
        video_start_token_id = hf_config.video_start_token_id
        video_end_token_id = hf_config.video_end_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        llm_pos_ids_list: list = []

        if not (image_grid_thw is None and video_grid_thw is None):
            if isinstance(image_grid_thw, Tensor):
                image_grid_thw = image_grid_thw.tolist()

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
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                    llm_pos_ids_list) > 0 else 0
                if modality_type == "image":
                    t, h, w = (
                        image_grid_thw[mm_data_idx][0],
                        image_grid_thw[mm_data_idx][1],
                        image_grid_thw[mm_data_idx][2],
                    )
                    llm_grid_t, llm_grid_h, llm_grid_w = \
                        t, h // spatial_merge_size, w // spatial_merge_size

                    t_index = mint.arange(llm_grid_t).view(-1, 1).expand(
                        -1, llm_grid_h * llm_grid_w).flatten()
                    h_index = mint.arange(llm_grid_h).view(1, -1, 1).expand(
                        llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = mint.arange(llm_grid_w).view(1, 1, -1).expand(
                        llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(
                        mint.stack([t_index, h_index, w_index]) + st_idx)
                    mm_data_idx += 1

                elif modality_type == "video":
                    t, h, w = (
                        video_frame_num,
                        image_grid_thw[mm_data_idx][1],
                        image_grid_thw[mm_data_idx][2],
                    )
                    llm_grid_t, llm_grid_h, llm_grid_w = \
                        t, h // spatial_merge_size, w // spatial_merge_size

                    for t_idx in range(llm_grid_t):
                        t_index = tensor(t_idx).view(-1, 1).expand(
                            -1, llm_grid_h * llm_grid_w).flatten()
                        h_index = mint.arange(llm_grid_h).view(
                            1, -1, 1).expand(1, -1, llm_grid_w).flatten()
                        w_index = mint.arange(llm_grid_w).view(
                            1, 1, -1).expand(1, llm_grid_h, -1).flatten()
                        llm_pos_ids_list.append(
                            mint.stack([t_index, h_index, w_index]) + st_idx)

                    mm_data_idx += 1
                    video_frame_num += 1

                else:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        mint.arange(text_len).view(1, -1).expand(3, -1) +
                        st_idx)
                    video_frame_num = 1

        else:
            text_len = len(input_tokens)
            llm_pos_ids_list.append(
                mint.arange(text_len).view(1, -1).expand(3, -1))

        llm_positions = mint.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        llm_positions = llm_positions[:, context_len:seq_len]
        mrope_position_delta = (llm_positions.max() + 1 -
                                len(input_tokens)).item()
        return llm_positions, mrope_position_delta

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

        if isinstance(image_grid_thw, mindspore.Tensor):
            image_grid_thw = image_grid_thw.tolist()
        if isinstance(video_grid_thw, mindspore.Tensor):
            video_grid_thw = video_grid_thw.tolist()

        input_tokens_tensor = mindspore.Tensor(input_tokens)
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
    def get_next_input_positions_tensor(
        mrope_position_delta: int,
        context_len: int,
        seq_len: int,
    ) -> mindspore.Tensor:
        return mint.arange(
            int(mrope_position_delta + context_len),
            int(mrope_position_delta + seq_len),
        ).broadcast_to((3, -1))


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
        dtype: mindspore.Type,
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

        self.mrope_section = mrope_section
        if self.mrope_section:
            assert sum(self.mrope_section) == rotary_dim // 2

    def construct(  # type: ignore[override]
        self,
        positions: mindspore.Tensor,
        query: mindspore.Tensor,
        key: mindspore.Tensor,
        batch_valid_length: Tensor = None,
    ) -> tuple[mindspore.Tensor, mindspore.Tensor]:
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
                if is_neox_style:
                    rotary_emb = InferMRotaryEmbedding(
                        head_size,
                        rotary_dim,
                        max_position,
                        base,
                        is_neox_style,
                        dtype,
                        mrope_section=rope_scaling["mrope_section"],
                    )
                else:
                    rotary_emb = MRotaryEmbedding(
                        head_size,
                        rotary_dim,
                        max_position,
                        base,
                        is_neox_style,
                        dtype,
                        mrope_section=rope_scaling["mrope_section"],
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
        else:
            raise NotImplementedError

    _ROPE_DICT[key] = rotary_emb  # type: ignore[assignment]
    return rotary_emb
