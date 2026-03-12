# SPDX-License-Identifier: Apache-2.0
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/phi4mm_audio.py
#
# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Copyright 2025 The vLLM Team.
# Copyright 2025 Microsoft Corporation.
# All rights reserved.
#
# This file is adapted from the vLLM implementation of the Phi-4 multimodal audio module.
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
"""Audio processing modules for Phi-4-multimodal-instruct model compatible with HuggingFace weights."""
# type: ignore
# isort:skip_file

import abc
import math
from typing import Any, Literal, Optional, Union

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
import mindspore.mint as mint
import mindspore.mint.nn.functional as F

from transformers import PretrainedConfig

from .phi4mm_utils import (
    AbsolutePositionalEncoding, ConvModule, FeedForward, MeanVarianceNormLayer,
    MultiHeadedAttention, MultiSequential, NemoConvSubsampling,
    T5RelativeAttentionLogitBias, adaptive_enc_mask, get_offset, unfold_tensor)

_AUDIO_PLACEHOLDER_TOKEN_ID = 200011  # <|endoftext11|>


class ConformerEncoderLayer(nn.Cell): 
    """ConformerEncoder Layer module (MindSpore版)."""

    def __init__(
        self,
        d_model: int = 512,
        ext_pw_out_channel: int = 0,
        depthwise_seperable_out_channel: int = 256,
        depthwise_multiplier: int = 1,
        n_head: int = 4,
        d_ffn: int = 2048,
        ext_pw_kernel_size: int = 1,
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
        causal: bool = False,
        batch_norm: bool = False,
        activation: str = "relu",
        chunk_se: int = 0,
        chunk_size: int = 18,
        conv_activation: str = "relu",
        conv_glu_type: str = "sigmoid",
        bias_in_glu: bool = True,
        linear_glu_in_convm: bool = False,
        attention_inner_dim: int = -1,
        attention_glu_type: str = "swish",
        activation_checkpointing: str = "",
        export: bool = False,
        use_pt_scaled_dot_product_attention: bool = False,
        attn_group_sizes: int = 1,
    ) -> None:
        super().__init__()
        self.feed_forward_in = FeedForward(
            d_model=d_model,
            d_inner=d_ffn,
            dropout_rate=dropout_rate,
            activation=activation,
            bias_in_glu=bias_in_glu,
        )
        self.self_attn = MultiHeadedAttention(
            n_head,
            d_model,
            dropout_rate,
            attention_inner_dim,
            attention_glu_type,
            bias_in_glu,
            use_pt_scaled_dot_product_attention=use_pt_scaled_dot_product_attention,
            group_size=attn_group_sizes,
        )
        self.conv = ConvModule(
            d_model,
            ext_pw_out_channel,
            depthwise_seperable_out_channel,
            ext_pw_kernel_size,
            kernel_size,
            depthwise_multiplier,
            dropout_rate,
            causal,
            batch_norm,
            chunk_se,
            chunk_size,
            conv_activation,
            conv_glu_type,
            bias_in_glu,
            linear_glu_in_convm,
            export=export,
        )
        self.feed_forward_out = FeedForward(
            d_model=d_model,
            d_inner=d_ffn,
            dropout_rate=dropout_rate,
            activation=activation,
            bias_in_glu=bias_in_glu,
        )
        self.layer_norm_att = mint.nn.LayerNorm([d_model])  
        self.layer_norm = mint.nn.LayerNorm([d_model])  

    def construct(
        self,
        x: ms.Tensor,
        pos_k: ms.Tensor,
        pos_v: ms.Tensor,
        mask: ms.Tensor,
        relative_attention_bias: Optional[ms.Tensor] = None,
    ) -> tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:  
        """
        x: (batch, max_time_in, size)
        pos_k, pos_v: 位置编码
        mask: (batch, max_time_in)
        relative_attention_bias: (1, n_head, time1, time2)
        """
        x = x + 0.5 * self.feed_forward_in(x)
        norm_x = self.layer_norm_att(x)
        x = x + self.self_attn(
            norm_x,
            norm_x,
            norm_x,
            pos_k,
            pos_v,
            mask,
            relative_attention_bias=relative_attention_bias,
        )
        x = x + self.conv(x)
        x = x + 0.5 * self.feed_forward_out(x)
        out = self.layer_norm(x)
        return out, pos_k, pos_v, mask

class TransformerEncoderBase(abc.ABC, nn.Cell): 
    """The Base class for Transformer based encoders (MindSpore版)"""

    def __init__(
        self,
        input_size: int,
        chunk_size: Union[int, list[int]],
        left_chunk: Union[int, list[int]],
        attention_dim: int = 256,
        attention_heads: int = 4,
        input_layer: str = "nemo_conv",
        cnn_out: int = -1,
        cnn_layer_norm: bool = False,
        time_reduction: int = 4,
        dropout_rate: float = 0.0,
        padding_idx: int = -1,
        relative_attention_bias_args: Optional[dict] = None,
        positional_dropout_rate: float = 0.0,
        nemo_conv_settings: Optional[dict] = None,
        conv2d_extra_padding: Literal["feat", "feat_time", "none", True] = "none",
        attention_group_size: int = 1,
        encoder_embedding_config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.input_layer = input_layer
        self.chunk_size = chunk_size
        self.left_chunk = left_chunk
        self.attention_dim = attention_dim
        self.num_heads = attention_heads
        self.attention_group_size = attention_group_size
        self.time_reduction = time_reduction
        self.nemo_conv_settings = nemo_conv_settings
        self.encoder_embedding_config = encoder_embedding_config

        if self.input_layer == "nemo_conv":
            default_nemo_conv_settings = {
                "subsampling": "dw_striding",
                "subsampling_factor": self.time_reduction,
                "feat_in": input_size,
                "feat_out": attention_dim,
                "conv_channels": 256,
                "subsampling_conv_chunking_factor": 1,
                "activation": nn.ReLU(),
                "is_causal": False,
            }
            if nemo_conv_settings:
                default_nemo_conv_settings.update(nemo_conv_settings)
                for i in ["subsampling_factor", "feat_in", "feat_out"]:
                    assert (
                        i not in nemo_conv_settings
                    ), f"{i} should be specified outside of the NeMo dictionary"
            self.embed = NemoConvSubsampling(**default_nemo_conv_settings)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.pos_emb = AbsolutePositionalEncoding(attention_dim, positional_dropout_rate)

        self.relative_attention_bias_type = (
            relative_attention_bias_args.get("type")
            if relative_attention_bias_args else None)
        if self.relative_attention_bias_type == "t5":
            assert (self.num_heads % self.attention_group_size == 0
                    ), "attention_group_size must divide n_head"
            self.relative_attention_bias_layer = T5RelativeAttentionLogitBias(
                self.num_heads // self.attention_group_size,
                max_distance=relative_attention_bias_args.get("t5_bias_max_distance", 1000),
                symmetric=relative_attention_bias_args.get("t5_bias_symmetric", False),
            )
        else:
            raise NotImplementedError

        self.encoder_embedding = MeanVarianceNormLayer(self.encoder_embedding_config["input_size"])

    def compute_lens_change(
        self,
        feature_lens: Union[int, ms.Tensor]
    ) -> Union[int, ms.Tensor]:
        """feature_lens: int or ms.Tensor, return updated feature lens."""
        if self.input_layer == "nemo_conv":
            subsampling_causal_cond = self.nemo_conv_settings.get(
                "subsampling", "dw_striding") in [
                    "dw_striding", "striding", "striding_conv1d",
                ]
            is_causal = self.nemo_conv_settings.get("is_causal", False)
            if is_causal and subsampling_causal_cond:
                if isinstance(feature_lens, ms.Tensor):
                    lens_change = mint.ceil(feature_lens / self.time_reduction).astype(ms.int64)
                    feature_lens_remainder = feature_lens % self.time_reduction
                    lens_change = ops.select(feature_lens_remainder != 1, lens_change + 1, lens_change)
                else:
                    lens_change = int(math.ceil(feature_lens / self.time_reduction))
                    feature_lens_remainder = feature_lens % self.time_reduction
                    if feature_lens_remainder != 1:
                        lens_change += 1
                return lens_change
            if isinstance(feature_lens, int):
                return int(math.ceil(feature_lens / self.time_reduction))
            else:
                return mint.ceil(feature_lens / self.time_reduction).astype(ms.int64)

    @abc.abstractmethod
    def forward(self) -> Any:
        """Abstract forward method implementation."""


    def _chunk_size_selection(
        self,
        chunk_size: Optional[Union[int, list[int]]] = None,
        left_chunk: Optional[Union[int, list[int]]] = None
    ) -> tuple[int, int]:
        """If chunk size is a list, randomly select a chunk size."""
        if chunk_size is None:
            chunk_size = self.chunk_size
        if left_chunk is None:
            left_chunk = self.left_chunk
        if isinstance(chunk_size, list):
            chunk_size_index = int(mint.randint(low=0, high=len(chunk_size), size=(1, )))
            chunk_size_train_eff = chunk_size[chunk_size_index]
            if not isinstance(left_chunk, list):
                raise ValueError("Since chunk_size is a list, left_chunk must be a list")
            if len(left_chunk) != len(chunk_size):
                raise ValueError("The length of left_chunk must be the same as length of chunk_size.")
            left_chunk_train_eff = left_chunk[chunk_size_index]
        else:
            chunk_size_train_eff = chunk_size
            left_chunk_train_eff = left_chunk
        return chunk_size_train_eff, left_chunk_train_eff


    def _forward_embeddings_core(
        self, input_tensor: ms.Tensor, masks: ms.Tensor
    ) -> tuple[ms.Tensor, ms.Tensor]:
        
        input_tensor, masks = self.embed(input_tensor, masks)
        return input_tensor, masks

    def _position_embedding(
        self, input_tensor: ms.Tensor
    ) -> tuple[Optional[ms.Tensor], Optional[ms.Tensor]]:
        pos_k = None
        pos_v = None
        if self.relative_attention_bias_layer is None:
            input_tensor = self.pos_emb(input_tensor)
        return pos_k, pos_v

    def _streaming_mask(
        self, seq_len: int, batch_size: int,
        chunk_size: Union[int, list[int]],
        left_chunk: Union[int, list[int]]
    ) -> ms.Tensor:
        chunk_size_train_eff, left_chunk_train_eff = self._chunk_size_selection(chunk_size, left_chunk)
        chunk_start_idx = np.arange(0, seq_len, chunk_size_train_eff)
        enc_streaming_mask = adaptive_enc_mask(
            seq_len, chunk_start_idx, left_window=left_chunk_train_eff
        )
        enc_streaming_mask = mint.unsqueeze(enc_streaming_mask, 0)
        enc_streaming_mask = ops.broadcast_to(enc_streaming_mask, (batch_size,) + enc_streaming_mask.shape[1:])
        return enc_streaming_mask

    def forward_embeddings(
        self,
        xs_pad: ms.Tensor,
        masks: ms.Tensor,
        chunk_size_nc: Optional[Union[int, list[int]]] = None,
        left_chunk_nc: Optional[Union[int, list[int]]] = None
    )-> Union[tuple[ms.Tensor, Optional[ms.Tensor],
                     Optional[ms.Tensor], ms.Tensor, ms.Tensor],
               tuple[ms.Tensor, Optional[ms.Tensor],
                     Optional[ms.Tensor], ms.Tensor, ms.Tensor,
                     ms.Tensor]]:
        seq_len = int(self.compute_lens_change(xs_pad.shape[1]))
        if seq_len <= 0:
            raise ValueError(
                f"The sequence length after time reduction is invalid: {seq_len}. "
                "Your input feature is too short. Consider filtering out the very short sentence from data loader"
            )
        batch_size = xs_pad.shape[0]
        enc_streaming_mask = self._streaming_mask(seq_len, batch_size, self.chunk_size, self.left_chunk)
        input_tensor = xs_pad
        input_tensor, masks = self._forward_embeddings_core(input_tensor, masks)
        streaming_mask = enc_streaming_mask
        if streaming_mask is not None and masks is not None:
            hs_mask = ops.logical_and(masks, streaming_mask)
        elif masks is not None:
            hs_mask = masks
        else:
            hs_mask = streaming_mask

        if chunk_size_nc is not None:
            enc_streaming_mask_nc = self._streaming_mask(seq_len, batch_size, chunk_size_nc, left_chunk_nc)
            if masks is not None:
                hs_mask_nc = ops.logical_and(masks, enc_streaming_mask_nc)
            else:
                hs_mask_nc = enc_streaming_mask_nc
        else:
            hs_mask_nc = None

        pos_k, pos_v = self._position_embedding(input_tensor)
        if chunk_size_nc is None:
            return input_tensor, pos_k, pos_v, hs_mask, masks
        return input_tensor, pos_k, pos_v, hs_mask, masks, hs_mask_nc

    def get_offset(self) -> int:
        return get_offset(self.input_layer, self.time_reduction)


class ConformerEncoder(TransformerEncoderBase): 

    extra_multi_layer_output_idxs: list[int]

    def __init__(
        self,
        input_size: int,
        chunk_size: Union[int, list[int]],
        left_chunk: Union[int, list[int]],
        num_lang: Optional[int] = None,
        attention_dim: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        input_layer: str = "nemo_conv",
        causal: bool = True,
        batch_norm: bool = False,
        cnn_out: int = -1,
        cnn_layer_norm: bool = False,
        ext_pw_out_channel: int = 0,
        ext_pw_kernel_size: int = 1,
        depthwise_seperable_out_channel: int = 256,
        depthwise_multiplier: int = 1,
        chunk_se: int = 0,
        kernel_size: int = 3,
        activation: str = "relu",
        conv_activation: str = "relu",
        conv_glu_type: str = "sigmoid",
        bias_in_glu: bool = True,
        linear_glu_in_convm: bool = False,
        attention_glu_type: str = "swish",
        export: bool = False,
        extra_layer_output_idx: int = -1,
        extra_multi_layer_output_idxs: list[int] = [],
        activation_checkpointing: str = "",
        relative_attention_bias_args: Optional[dict] = None,
        time_reduction: int = 4,
        use_pt_scaled_dot_product_attention: bool = False,
        nemo_conv_settings: Optional[dict] = None,
        conv2d_extra_padding: Literal["feat", "feat_time", "none", True] = "none",
        replication_pad_for_subsample_embedding: bool = False,
        attention_group_size: int = 1,
        encoder_embedding_config: Optional[dict] = None,
    ) -> None:
        super().__init__(
            input_size,
            chunk_size,
            left_chunk,
            attention_dim,
            attention_heads,
            input_layer,
            cnn_out,
            cnn_layer_norm,
            time_reduction,
            dropout_rate=dropout_rate,
            relative_attention_bias_args=relative_attention_bias_args,
            positional_dropout_rate=0.0,
            nemo_conv_settings=nemo_conv_settings,
            conv2d_extra_padding=conv2d_extra_padding,
            attention_group_size=attention_group_size,
            encoder_embedding_config=encoder_embedding_config,
        )
        self.num_blocks = num_blocks
        self.num_lang = num_lang
        self.kernel_size = kernel_size
        self.replication_pad_for_subsample_embedding: bool = replication_pad_for_subsample_embedding
        assert (self.num_heads % attention_group_size == 0), "attention_group_size must divide n_head"
        self.num_heads_k = self.num_heads // attention_group_size

        self.encoders = MultiSequential(*[
            ConformerEncoderLayer(
                d_model=attention_dim,
                ext_pw_out_channel=ext_pw_out_channel,
                depthwise_seperable_out_channel=depthwise_seperable_out_channel,
                depthwise_multiplier=depthwise_multiplier,
                n_head=attention_heads,
                d_ffn=linear_units,
                ext_pw_kernel_size=ext_pw_kernel_size,
                kernel_size=kernel_size,
                dropout_rate=dropout_rate,
                causal=causal,
                batch_norm=batch_norm,
                activation=activation,
                chunk_se=chunk_se,
                chunk_size=chunk_size,
                conv_activation=conv_activation,
                conv_glu_type=conv_glu_type,
                bias_in_glu=bias_in_glu,
                linear_glu_in_convm=linear_glu_in_convm,
                attention_glu_type=attention_glu_type,
                activation_checkpointing=activation_checkpointing,
                export=export,
                use_pt_scaled_dot_product_attention=use_pt_scaled_dot_product_attention,
                attn_group_sizes=attention_group_size,
            ) for _ in range(num_blocks)
        ])
        self.extra_layer_output_idx = extra_layer_output_idx
        self.extra_multi_layer_output_idxs = extra_multi_layer_output_idxs
        
        self.dev_type = ms.Parameter(ms.Tensor(0, dtype=ms.float32), requires_grad=False)

    def init_relative_attention_bias(self, input_tensor: ms.Tensor) -> Optional[ms.Tensor]:
        if self.relative_attention_bias_layer:
            return self.relative_attention_bias_layer(input_tensor)

    def calculate_hs_mask(self, xs_pad: ms.Tensor, mask: Optional[ms.Tensor]) -> ms.Tensor:
        max_audio_length = xs_pad.shape[1]
        batch_size = xs_pad.shape[0]
        enc_streaming_mask = self._streaming_mask(max_audio_length, batch_size, self.chunk_size, self.left_chunk)
        if mask is None:
            return enc_streaming_mask

        feature_lens = mask.sum(1)
        padding_length = feature_lens
        
        pad_mask = mint.arange(0, max_audio_length).expand_dims(0).broadcast_to((padding_length.shape[0], max_audio_length))
        pad_mask = pad_mask < mint.unsqueeze(padding_length, 1)
        pad_mask = mint.unsqueeze(pad_mask, 1)
        pad_mask = ops.logical_and(pad_mask, enc_streaming_mask)
        return pad_mask

    def construct(self, xs_pad: ms.Tensor, masks: ms.Tensor) -> tuple[ms.Tensor, ms.Tensor]:  
        xs_pad = self.encoder_embedding(xs_pad)
        input_tensor, pos_k, pos_v, hs_mask, masks = self.forward_embeddings(xs_pad, masks)

        unfolded = False
        ori_bz, seq_len, D = input_tensor.shape
        max_seq_len = 500
        if seq_len > max_seq_len:
            unfolded = True
            chunk_pad_size = max_seq_len - (seq_len % max_seq_len) if seq_len % max_seq_len > 0 else 0
            if chunk_pad_size > 0:
                
                input_tensor_pad = F.pad(input_tensor,
                                         (0, 0, 0, chunk_pad_size), "constant",
                                         0)
            input_tensor = unfold_tensor(input_tensor, max_seq_len)
            if masks is not None:
                subsampled_pad_mask = mint.squeeze(masks, 1)
                extra_padded_subsamlped_pad_mask = F.pad(
                    subsampled_pad_mask, (0, chunk_pad_size), "constant",
                    False) 
                extra_padded_subsamlped_pad_mask = mint.unsqueeze(extra_padded_subsamlped_pad_mask, -1).astype(ms.float32)
                masks_unfold = unfold_tensor(extra_padded_subsamlped_pad_mask, max_seq_len)
                masks_unfold = mint.squeeze(masks_unfold, -1).astype(ms.bool_)
            else:
                masks_unfold = None
            hs_mask = self.calculate_hs_mask(input_tensor, masks_unfold)

        relative_attention_bias = self.init_relative_attention_bias(input_tensor)
        _simplified_path = (self.extra_layer_output_idx == -1 and relative_attention_bias is None)

        if _simplified_path:
            input_tensor, *_ = self.encoders(input_tensor, pos_k, pos_v, hs_mask)
        else:
            for i, layer in enumerate(self.encoders):
                input_tensor, _, _, _ = layer(
                    input_tensor,
                    pos_k,
                    pos_v,
                    hs_mask,
                    relative_attention_bias=relative_attention_bias,
                )

        if unfolded:
            embed_dim = input_tensor.shape[-1]
            input_tensor = mint.reshape(input_tensor, (ori_bz, -1, embed_dim))
            if chunk_pad_size > 0:
                input_tensor = input_tensor[:, :-chunk_pad_size, :]

        return input_tensor, masks  # , layer_emb


class WindowQformer(nn.Cell): 
    """Window-level Qformer (MindSpore版)"""

    def __init__(
        self,
        window_size: int = 8,
        num_queries: int = 1,
        num_blocks: int = 2,
        attention_dim: int = 512,
        attention_heads: int = 8,
        linear_units: int = 2048,
        dropout_rate: float = 0.0,
        normalize_before: bool = True,
    ):
        super().__init__()
        
        self.decoders = nn.CellList([
            nn.TransformerDecoderLayer(
                d_model=attention_dim,
                nhead=attention_heads,
                dim_feedforward=linear_units,
                dropout=dropout_rate,
                activation="relu",
                batch_first=True,
                norm_first=normalize_before,  
            ) for _ in range(num_blocks)
        ])
        self.queries = ms.Parameter(
            mint.zeros((1, num_queries, attention_dim), dtype=ms.float32), name="queries"
        )
        self.after_norm = mint.nn.LayerNorm((attention_dim,), eps=1e-12) if normalize_before else None
        self.window_size = window_size

    def construct(
        self,
        audio_embed: ms.Tensor,
        mask: Optional[ms.Tensor],
        embed_len: Optional[int] = None
    ) -> tuple[ms.Tensor, Optional[int]]:
        # audio_embed: N x T x D => N x D x T
        audio_embed = mint.transpose(audio_embed, 1, 2)  
        padding = audio_embed.shape[-1] % self.window_size
        if padding > 0:
            audio_embed = F.pad(audio_embed, (0, self.window_size - padding),
                                "constant", 0)

        embed_chunk = F.unfold(
            audio_embed[..., None, :],
            kernel_size=(1, self.window_size),
            stride=(1, self.window_size),
        )
        bsz, _, slen = embed_chunk.shape
        embed_chunk = embed_chunk.view((bsz, -1, self.window_size, slen))
        embed_chunk = mint.transpose(embed_chunk, 1, 3)  
        embed_chunk = embed_chunk.view((bsz * slen, self.window_size, -1))
        q = ops.broadcast_to(self.queries, (bsz * slen, self.queries.shape[1], self.queries.shape[2]))
        for layer in self.decoders:
            q = layer(q, embed_chunk, tgt_mask=None, memory_mask=mask)

        if self.after_norm is not None:
            q = self.after_norm(q)

        if embed_len is not None:
            embed_len = embed_len // self.window_size
        out = q.view((bsz, slen, -1))
        return out, embed_len

class AudioEmbedding(nn.Cell): 

    def __init__(self, config: PretrainedConfig, **kwargs: Any) -> None:
        super().__init__()
        self.config = config
        hidden_size = (config.n_embd if hasattr(config, "n_embd") else config.hidden_size)

        audio_dim_out = None
        self.layer_idx = -2

        if (isinstance(config.audio_processor, dict)
                and config.audio_processor.get("name", None) == "cascades"):
            encoder_config = config.audio_processor.get("config", None)
            assert encoder_config is not None
            self.encoder = ConformerEncoder(**encoder_config) 
            audio_dim_out = encoder_config["attention_dim"]
            n_mels = encoder_config["input_size"]
        else:
            raise NotImplementedError("")

        assert (audio_dim_out is not None), "Remember to set values for audio_dim_out"
        self.audio_dim_out = audio_dim_out
        self.audio_dim_in = n_mels

        self.freeze_audio_processor = kwargs.get("freeze_audio_processor", False)
        self.downsample_rate = kwargs.get("downsample_rate", 1)

        if kwargs.get("use_qformer", False):
            qformer_config = kwargs.get("qformer_config", {})
            qformer_config["attention_dim"] = audio_dim_out
            self.qformer = WindowQformer(**qformer_config)  
        else:
            self.qformer = None

        if kwargs.get("use_conv_downsample", False):
            assert (self.qformer is None), "don't support use qformer and conv downsample together"
            nemo_conv_settings = kwargs.get("nemo_conv_settings", {})
            default_nemo_conv_settings = {
                "subsampling": "dw_striding",
                "subsampling_factor": self.downsample_rate,
                "feat_in": audio_dim_out,
                "feat_out": audio_dim_out,
                "conv_channels": 256,
                "subsampling_conv_chunking_factor": 1,
                "activation": nn.ReLU(),
                "is_causal": False,
            }
            if nemo_conv_settings:
                default_nemo_conv_settings.update(nemo_conv_settings)
                for i in ["subsampling_factor", "feat_in", "feat_out"]:
                    assert (i not in nemo_conv_settings), "{i} should be specified outside of the NeMo dictionary"
            self.conv_ds = NemoConvSubsampling(**default_nemo_conv_settings)  
        else:
            self.conv_ds = None

        projection_cls = kwargs.get("projection_cls", "linear")
        if projection_cls == "linear":
            self.audio_projection = mint.nn.Linear(audio_dim_out, hidden_size) 
        elif projection_cls == "mlp":
            dim_projection = hidden_size
            depth = 2
            self.linear_downsample_rate = (1 if (self.qformer or self.conv_ds) else self.downsample_rate)
            layers = [
                mint.nn.Linear(audio_dim_out * self.linear_downsample_rate, dim_projection)
            ]
            for _ in range(1, depth):
                layers.extend([
                    nn.GELU(),
                    mint.nn.Linear(dim_projection, dim_projection)
                ])
            self.audio_projection = nn.SequentialCell(layers)  

            layers = [
                mint.nn.Linear(audio_dim_out * self.linear_downsample_rate, dim_projection)
            ]
            for _ in range(1, depth):
                layers.extend([
                    nn.GELU(),
                    mint.nn.Linear(dim_projection, dim_projection)
                ])
            self.audio_projection_for_vision = nn.SequentialCell(layers)
        else:
            raise NotImplementedError(f"projection_cls = {projection_cls}, not implemented")

        self.vocab_size = config.vocab_size
        self.input_embeds = None
        self.audio_embed_sizes = None

    def set_audio_embeds(self, input_embeds: ms.Tensor) -> None: 
        self.input_embeds = input_embeds

    def set_audio_embed_sizes(self, audio_embed_sizes: ms.Tensor) -> None: 
        self.audio_embed_sizes = audio_embed_sizes

    def get_audio_features(
        self,
        input_embeds: ms.Tensor,
        audio_attention_mask: Optional[ms.Tensor] = None,
        audio_projection_mode: str = "speech",
    ) -> ms.Tensor:
        """
        arguments:
            input_embeds: audio features (B, T, D)  B: num audios in a sequence
        """
        if self.freeze_audio_processor:
            audio_features, masks = self.encoder(input_embeds, audio_attention_mask)
        else:
            audio_features, masks = self.encoder(input_embeds, audio_attention_mask)

        if self.qformer is not None:
            audio_features, _ = self.qformer(audio_features, mask=None)

        if self.conv_ds is not None:
            if masks is not None:
                masks = mint.squeeze(masks, 1)
            audio_features, masks = self.conv_ds(audio_features, mask=masks)

        if hasattr(self, "linear_downsample_rate") and self.linear_downsample_rate != 1:
            bs, seq_len, feat_dim = audio_features.shape
            padding = seq_len % self.linear_downsample_rate
            if padding > 0:
               audio_features = F.pad(
                    audio_features,
                    (0, 0, 0, self.linear_downsample_rate - padding),
                    "constant",
                    0,
                )
            seq_len = audio_features.shape[1]
            audio_features = audio_features.view((bs, seq_len // self.linear_downsample_rate, feat_dim * self.linear_downsample_rate))

        if audio_projection_mode == 'speech':
            audio_set_tensor = self.audio_projection(audio_features)
        elif audio_projection_mode == 'vision':
            audio_set_tensor = self.audio_projection_for_vision(audio_features)
        else:
            raise ValueError(f"audio_projection_mode = {audio_projection_mode} not implemented")

        return audio_set_tensor

    def construct(
        self,
        audio_features: ms.Tensor,
        audio_attention_mask: Optional[ms.Tensor] = None,
        audio_projection_mode: str = "speech",
    ) -> ms.Tensor: 
        audio_embeds = self.get_audio_features(
            mint.unsqueeze(audio_features, 0),  
            audio_attention_mask=audio_attention_mask,
            audio_projection_mode=audio_projection_mode,
        )
        return mint.squeeze(audio_embeds, 0)



