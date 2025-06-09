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

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import Tensor, mint, ops, nn
from mindspore.common import dtype as mstype

from transformers import PretrainedConfig

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
        base: int,
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
        # self.register_buffer("cos_sin_cache", cache, persistent=False)

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

        # freqs = ops.einsum("i,j -> ij", t, inv_freq)
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
    ) -> Tuple[Tensor, Tensor]:
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
        base: int,
        is_neox_style: bool,
        dtype,
    ) -> None:
        if not is_neox_style:
            raise NotImplementedError(
                "InferRotaryEmbedding only support Neox-style rotary embeddings."
            )
        super().__init__()
        self.rotary_embedding_op = ops.ApplyRotaryPosEmb(2)
        self.gather = ops.Gather()
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
        freqs_base = np.arange(0, self.rotary_dim, 2).astype(
            np.float32)  # (head_dim // 2, )
        freqs = 1.0 / (base**(freqs_base / self.rotary_dim))  # (head_dim // 2, )
        return freqs

    def _compute_cos_sin_cache(self) -> Tuple[Tensor, Tensor]:
        freqs = self._compute_inv_freq(self.base)
        t = np.arange(0, self.max_position_embeddings, 1).astype(np.float32)
        freqs = np.outer(t, freqs)  # (max_position_embedding, head_dim // 2)
        emb = np.concatenate((freqs, freqs), axis=-1)
        freqs_cos = np.cos(emb) # (seq_len, head_dim)
        freqs_sin = np.sin(emb) # (seq_len, head_dim)
        freqs_cos = Tensor(freqs_cos, dtype=self.dtype)
        freqs_sin = Tensor(freqs_sin, dtype=self.dtype)
        return freqs_cos, freqs_sin

    def construct(
        self,
        positions: Tensor,
        query: Tensor,
        key: Tensor,
        batch_valid_length: Tensor,
        is_prefill: bool,
        offsets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        query = query.contiguous()
        key = key.contiguous()
        if is_prefill:
            return self.rotary_embedding_op(query, key, self.freqs_cos,
                                            self.freqs_sin, batch_valid_length)

        freqs_cos = self.gather(self.freqs_cos, positions, 0)
        freqs_sin = self.gather(self.freqs_sin, positions, 0)
        return self.rotary_embedding_op(query, key, freqs_cos, freqs_sin,
                                        batch_valid_length)


class MRotaryEmbedding(RotaryEmbedding):
    """Rotary Embedding with Multimodal Sections."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: mindspore.Type,
        mrope_section: Optional[List[int]] = None,
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
        is_prefill: bool = False,
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
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
        # cos_sin_cache: (4*max_pos, rotary_dim//2 * 2)
        # positions: (3, 5120)
        # cos_sin: (3, 5120, rotary_dim)
        # cos/sin: cat[(1, 5120, mrope_sec),...] -> (1, 5120, rotary_dim//2)
        ######################################################################
        num_tokens = positions.shape[-1]
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = ops.chunk(cos_sin, 2, axis=-1)
        if positions.ndim == 2:
            cos_l = ops.split(cos, self.mrope_section, axis=-1)
            sin_l = ops.split(sin, self.mrope_section, axis=-1)
            cos, sin = (), ()
            for i in range(len(self.mrope_section)):
                cos += (cos_l[i][i],)
                sin += (sin_l[i][i],)
            cos = ops.cat(cos, axis=-1)
            sin = ops.cat(sin, axis=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = ops.cat((query_rot, query_pass), axis=-1).view(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = ops.cat((key_rot, key_pass), axis=-1).view(key_shape)
        return query, key

    @staticmethod
    def get_input_positions(
        input_tokens: List[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[List[List[int]], mindspore.Tensor],
        video_grid_thw: Union[List[List[int]], mindspore.Tensor],
        second_per_grid_ts: Optional[List[float]] = None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
    ) -> Tuple[List[List[int]], int]:
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

    @staticmethod
    def get_input_positions_tensor(
        input_tokens: List[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[List[List[int]], mindspore.Tensor],
        video_grid_thw: Union[List[List[int]], mindspore.Tensor],
        second_per_grid_ts: Optional[List[float]] = None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
    ) -> Tuple[mindspore.Tensor, int]:
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
                ops.arange(text_len).view(1, -1).broadcast_to((3, -1)).int() + st_idx)

            t_index = (ops.arange(llm_grid_t).view(-1, 1).broadcast_to(
                (-1, llm_grid_h * llm_grid_w)) * video_second_per_grid_t *
                       tokens_per_second).int().flatten()
            h_index = ops.arange(llm_grid_h).view(1, -1, 1).broadcast_to(
                (llm_grid_t, -1, llm_grid_w)).flatten().int()
            w_index = ops.arange(llm_grid_w).view(1, 1, -1).broadcast_to(
                (llm_grid_t, llm_grid_h, -1)).flatten().int()
            
            llm_pos_ids_list.append(
                ops.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                ops.arange(text_len).view(1, -1).broadcast_to((3, -1)).int() + st_idx)

        llm_positions = ops.cat(llm_pos_ids_list, axis=1).view(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 -
                                len(input_tokens)).item()
        llm_positions = llm_positions[:, context_len:seq_len]

        return llm_positions, mrope_position_delta

    @staticmethod
    def get_next_input_positions(
        mrope_position_delta: int,
        context_len: int,
        seq_len: int,
    ) -> List[List[int]]:
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
        return ops.arange(
            mrope_position_delta + context_len,
            mrope_position_delta + seq_len,
        ).broadcast_to((3, -1))


class InferMRotaryEmbedding(InferRotaryEmbedding):
    """Rotary Embedding with Multimodal Sections."""

    get_input_positions = MRotaryEmbedding.get_input_positions
    get_input_positions_tensor = MRotaryEmbedding.get_input_positions_tensor
    get_next_input_positions = MRotaryEmbedding.get_next_input_positions
    get_next_input_positions_tensor = MRotaryEmbedding.get_next_input_positions_tensor

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: mindspore.Type,
        mrope_section: Optional[List[int]] = None,
    ) -> None:
        # In Qwen2.5-VL, the maximum index value is related to the duration of
        # the input video. We enlarge max_position_embeddings to 4 times to get
        # a larger the cos and sin cache.
        self.cache_max_position_num = max_position_embeddings * 4
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
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
        is_prefill: bool = False,
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """
        Args:
            positions:
                [num_tokens,] (text only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        # prefill
        if is_prefill:
            num_tokens = positions.shape[-1]
            cos, sin = self.freqs_cos[positions], self.freqs_sin[positions]
            cos, sin = cos[..., :self.rotary_dim//2], sin[..., :self.rotary_dim//2]
            if positions.ndim == 2:
                cos_l = ops.split(cos, self.mrope_section, axis=-1)
                sin_l = ops.split(sin, self.mrope_section, axis=-1)
                cos, sin = (), ()
                for i in range(len(self.mrope_section)):
                    cos += (cos_l[i][i],)
                    sin += (sin_l[i][i],)
                cos = ops.cat(cos, axis=-1)
                sin = ops.cat(sin, axis=-1)

            query_shape = query.shape
            query = query.view(num_tokens, -1, self.head_size)
            query_rot = query[..., :self.rotary_dim]
            query_pass = query[..., self.rotary_dim:]
            query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
            query = ops.cat((query_rot, query_pass), axis=-1).view(query_shape)

            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_size)
            key_rot = key[..., :self.rotary_dim]
            key_pass = key[..., self.rotary_dim:]
            key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
            key = ops.cat((key_rot, key_pass), axis=-1).view(key_shape)
            return query, key

        # decode
        if positions.ndim == 2 and positions.shape[0] == len(self.mrope_section):
            num_tokens = positions.shape[-1]
            cos, sin = self.freqs_cos[positions], self.freqs_sin[positions]
            cos, sin = cos[..., :self.rotary_dim//2], sin[..., :self.rotary_dim//2]
            cos_l = ops.split(cos, self.mrope_section, axis=-1)
            sin_l = ops.split(sin, self.mrope_section, axis=-1)
            cos, sin = (), ()
            for i in range(len(self.mrope_section)):
                cos += (cos_l[i][i],)
                sin += (sin_l[i][i],)
            cos = ops.cat(cos, axis=-1)
            sin = ops.cat(sin, axis=-1)
            freqs_cos = ops.cat([cos, cos], axis=-1).squeeze(1)
            freqs_sin = ops.cat([sin, sin], axis=-1).squeeze(1)
        else:
            positions = positions.flatten()
            freqs_cos = self.freqs_cos.index_select(0, positions)
            freqs_sin = self.freqs_sin.index_select(0, positions)

        return self.rotary_embedding_op(query, key, freqs_cos, freqs_sin, batch_valid_length)


_ROPE_DICT: Dict[Tuple, InferRotaryEmbedding] = {}


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[Any] = mstype.float16,
    partial_rotary_factor: float = 1.0,
) -> InferRotaryEmbedding:
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
            raise NotImplementedError
        elif scaling_type == "default":
            if "mrope_section" in rope_scaling:
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
                raise NotImplementedError
        else:
            raise NotImplementedError

    _ROPE_DICT[key] = rotary_emb
    return rotary_emb
