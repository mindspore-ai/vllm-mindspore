# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
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
"""Inference-only Qwen3-Omni-Moe model (thinker part)."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import Any

import math
import numpy as np
# import torch
from mindspore import Parameter, Tensor, mint, nn
import mindspore.mint.nn.functional as F
import mindspore as ms
from mindspore import ops
from mindspore.ops.operations.nn_ops import FlashAttentionScore
import ms_custom_ops
from packaging.version import Version
from transformers import PretrainedConfig
from transformers import __version__ as TRANSFORMERS_VERSION
from transformers.feature_extraction_utils import BatchFeature

from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeThinkerConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
)
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, \
    get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from vllm.distributed import utils as dist_utils
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm_mindspore.model_executor.layers.layernorm import RMSNorm
from vllm_mindspore.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm_mindspore.model_executor.layers.activation import SiluAndMul
from vllm_mindspore.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    PlaceholderFeaturesInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import uses_mrope
from vllm.model_executor.models.utils import init_vllm_registered_model
from vllm.config import get_current_vllm_config
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.transformers_utils.tokenizer import decode_tokens, encode_tokens

from vllm_mindspore.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
)
from vllm_mindspore.model_executor.models.utils import (WeightsMapper,
    maybe_prefix, merge_multimodal_embeddings)
from vllm_mindspore.model_executor.models.model_base import NativeModel, \
    AttentionWrapper
from vllm_mindspore.model_executor.models.attention_mask import \
    MultiModalLowerTriangularMask
from vllm_mindspore.model_executor.models.qwen2_5_vl import (
    _qwen2vl_field_config,
    Qwen2_5_VisionAttention,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VLProcessingInfo,
)
from vllm_mindspore.model_executor.models.qwen2_audio import (
    Qwen2AudioProcessingInfo,
    Qwen2AudioInputs,
)
from vllm_mindspore.utils import is_310p
from vllm_mindspore.model_executor.layers.logits_processor import (
    LogitsProcessor)
from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead)
from vllm_mindspore.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeModel
from vllm_mindspore.model_executor.models.vision import get_llm_pos_ids_for_vision
from vllm_mindspore.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniConditionalGenerationMixin,
    Qwen2_5OmniThinkerDummyInputsBuilder,
    Qwen2_5OmniThinkerMultiModalProcessor,
    Qwen2_5OmniThinkerProcessingInfo,
)

from mindspore.common.api import _pynative_executor
_pynative_executor.set_enable_grad(False)


logger = init_logger(__name__)


_ACTIVATION_REGISTRY = {"gelu_pytorch_tanh": mint.nn.GELU(approximate="tanh")}

def _get_feat_extract_output_lengths(input_lengths: ms.Tensor):
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    )
    return feat_lengths, output_lengths


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
        q = q.reshape(1, seq_length, self.num_attention_heads_per_partition,
                      self.hidden_size_per_attention_head)
        k = k.reshape(1, seq_length, self.num_attention_heads_per_partition,
                      self.hidden_size_per_attention_head)

        cos, sin = position_embeddings
        origin_dtype = q.dtype
        q, k = ms_custom_ops.apply_rotary_pos_emb_ext(q.astype(ms.float32),
                                                      k.astype(ms.float32),
                                                      cos, sin, "BSND", "half")

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


class Qwen3Omni_VisionTransformer(nn.Cell):
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
        rotary_pos_emb = rotary_pos_emb.reshape(1, seq_len, 1, -1)
        emb = mint.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
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
                if name in params_dict:
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


class Qwen3MoeLLMModel(Qwen3MoeModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        self.deepstack_multiscale_layer_start = 1

    def construct(
        self,
        input_ids: ms.Tensor,
        positions: ms.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: ms.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
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
        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer]
        ):
            layer_idx = layer_idx + self.start_layer

            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

            if deepstack_input_embeds is not None and layer_idx in range(
                0, len(deepstack_input_embeds)
            ):
                hidden_states = (
                    hidden_states
                    + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeLLMForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3MoeForCausalLM, self).__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3MoeLLMModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
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


class Qwen3OmniMoeThinkerProcessingInfo(
    Qwen2AudioProcessingInfo, Qwen2_5_VLProcessingInfo
):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3OmniMoeConfig).thinker_config

    def get_hf_processor(self, **kwargs: object) -> Qwen3OmniMoeProcessor:
        processor = self.ctx.get_hf_processor(
            Qwen3OmniMoeProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )
        if not hasattr(processor, "audio_token"):
            processor.audio_token = "<|audio_pad|>"
        if not hasattr(processor, "image_token"):
            processor.image_token = "<|image_pad|>"
        if not hasattr(processor, "video_token"):
            processor.video_token = "<|video_pad|>"
        return processor

    def get_feature_extractor(self, **kwargs: object):
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None, "image": None, "video": None}



Qwen3OmniMoeThinkerDummyInputsBuilder = Qwen2_5OmniThinkerDummyInputsBuilder


class Qwen3OmniMoeThinkerMultiModalProcessor(
    Qwen2_5OmniThinkerMultiModalProcessor,
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        # tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])

        def pad_to_hop_length(x: np.ndarray, hop_length: int) -> np.ndarray:
            length = x.shape[-1]
            if length % hop_length != 0:
                pad_length = hop_length - (length % hop_length)
                x = np.pad(x, (0, pad_length), mode="constant", constant_values=0)
            return x

        # NOTE: WhisperFeatureExtractor cannot handle empty list of audios
        feature_extractor = self.info.get_feature_extractor()
        hop_length = feature_extractor.hop_length
        if audios:
            # NOTE: Qwen3-Omni processor accept "audio"
            # To make sure the cache works with padding=True, we pre-padded
            # the audio to multiple of hop_length.
            mm_data["audio"] = [
                pad_to_hop_length(audio, hop_length)
                if isinstance(audio, np.ndarray)
                else (pad_to_hop_length(audio[0], hop_length), audio[1])
                for audio in audios
            ]
            mm_kwargs = dict(
                **mm_kwargs,
            )
            # TODO(Isotr0py): Remove this patch after upstream fix PR
            # released and Transformers version update:
            # https://github.com/huggingface/transformers/pull/41473
            if (
                Version(TRANSFORMERS_VERSION) < Version("4.58.0")
                and "truncation" not in mm_kwargs
            ):
                mm_kwargs["truncation"] = False

        hf_inputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            # tok_kwargs=tok_kwargs,
        )

        if (
            "audio_feature_lengths" in hf_inputs
            and "feature_attention_mask" in hf_inputs
            and (audios := mm_data.get("audio", []))
        ):
            audio_num_frames = []
            for _, audio in enumerate(audios):
                audio_length = len(audio[0]) if isinstance(audio, tuple) else len(audio)
                num_frame = (
                    (audio_length // hop_length)
                    if audio_length % hop_length == 0
                    else (audio_length // hop_length - 1)
                )
                if mm_kwargs.get("truncation", False):
                    num_frame = min(
                        num_frame, feature_extractor.n_samples // hop_length
                    )
                audio_num_frames.append(num_frame)
            hf_inputs["feature_attention_mask"] = [
                mint.ones(num_frame) for num_frame in audio_num_frames
            ]
            hf_inputs["audio_feature_lengths"] = ms.Tensor(audio_num_frames)
        return hf_inputs

    # def _maybe_apply_prompt_updates(
    #     self,
    #     mm_items: MultiModalDataItems,
    #     prompt_ids: list[int],
    #     mm_kwargs: MultiModalKwargs,
    #     mm_prompt_updates: MultiModalPromptUpdates,
    #     is_update_applied: bool,
    # ) -> tuple[list[int], str, Mapping[str, list[PlaceholderFeaturesInfo]]]:
    #     """
    #     Qwen3-Omni reimplements this function to handle `use_audio_in_video`.
    #     """
    #     mm_item_counts = mm_items.get_all_counts()
    #     self._validate_mm_kwargs(mm_kwargs, mm_item_counts)

    #     use_audio_in_video = False
    #     if "video" in mm_kwargs:
    #         for item in mm_kwargs["video"]:
    #             if item and item["use_audio_in_video"].data:
    #                 use_audio_in_video = True
    #             else:
    #                 use_audio_in_video = False

    #     if use_audio_in_video and "video" in mm_item_counts:
    #         assert "audio" in mm_item_counts
    #         mm_item_counts["audio"] -= mm_item_counts["video"]

    #     # Special case with `use_audio_in_video=True`
    #     if use_audio_in_video:
    #         if is_update_applied:
    #             prompt_ids = self._get_raw_input_ids(prompt_ids, use_audio_in_video)
    #         (
    #             prompt_ids,
    #             mm_placeholders,
    #         ) = self._apply_prompt_updates(
    #             prompt_ids,
    #             mm_prompt_updates,
    #         )
    #         self._validate_mm_placeholders(mm_placeholders, mm_item_counts)
    #     # normal case with `use_audio_in_video=False`
    #     elif is_update_applied:
    #         mm_placeholders = self._find_mm_placeholders(
    #             prompt_ids,
    #             mm_prompt_updates,
    #         )
    #         self._validate_mm_placeholders(
    #             mm_placeholders,
    #             mm_item_counts,
    #         )
    #     else:
    #         prompt_ids, mm_placeholders = self._apply_prompt_updates(
    #             prompt_ids,
    #             mm_prompt_updates,
    #         )
    #         self._validate_mm_placeholders(
    #             mm_placeholders,
    #             mm_item_counts,
    #         )

    #     return prompt_ids, mm_placeholders

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargs,
        is_update_applied: bool,
    ) -> tuple[list[int], str, Mapping[str, list[PlaceholderFeaturesInfo]]]:
        unbound_prompt_updates = self._get_prompt_updates(
            mm_items,
            hf_processor_mm_kwargs,
            mm_kwargs,
        )
        mm_prompt_updates = self._bind_and_group_updates(
            unbound_prompt_updates)

        mm_item_counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_kwargs, mm_item_counts)

        use_audio_in_video = False
        if "video" in mm_kwargs:
            for item in mm_kwargs["video"]:
                if item and item["use_audio_in_video"].data:
                    use_audio_in_video = True
                else:
                    use_audio_in_video = False

        if use_audio_in_video and "video" in mm_item_counts:
            assert "audio" in mm_item_counts
            mm_item_counts["audio"] -= mm_item_counts["video"]

        # Special case with `use_audio_in_video=True`
        if use_audio_in_video:
            if is_update_applied:
                prompt_ids = self._get_raw_input_ids(prompt_ids, use_audio_in_video)
            (
                prompt_ids,
                prompt,
                mm_placeholders,
            ) = self._apply_prompt_updates(
                prompt_ids,
                mm_prompt_updates,
                mm_item_counts
            )
            self._validate_mm_placeholders(mm_placeholders, mm_item_counts)
        # normal case with `use_audio_in_video=False`
        elif is_update_applied:
            mm_placeholders = self._find_mm_placeholders(
                mm_prompt_updates,
                prompt_ids,
                mm_item_counts
            )
            self._validate_mm_placeholders(
                mm_placeholders,
                mm_item_counts,
            )
        else:
            prompt_ids, mm_placeholders = self._apply_prompt_updates(
                prompt_ids,
                mm_prompt_updates,
                mm_item_counts
            )
            self._validate_mm_placeholders(
                mm_placeholders,
                mm_item_counts,
            )
        tokenizer = self.info.get_tokenizer()
        prompt = decode_tokens(tokenizer, prompt_ids)
        return prompt_ids, prompt, mm_placeholders

    def get_updates_use_audio_in_video(
        self,
        thinker_config: PretrainedConfig,
        audio_len: int,
        video_grid_thw: list[int] | ms.Tensor,
        video_second_per_grid_t: float,
    ) -> list[int]:
        shift = 0
        audio_token_id = thinker_config.audio_token_id
        video_token_id = thinker_config.video_token_id
        audio_start_token_id = thinker_config.audio_start_token_id
        audio_end_token_id = thinker_config.audio_end_token_id
        spatial_merge_size = thinker_config.vision_config.spatial_merge_size
        position_id_per_seconds = thinker_config.position_id_per_seconds
        audio_token_indices = np.arange(next(iter([audio_len])))
        curr_video_grid_thw = next(iter([video_grid_thw]))
        height = curr_video_grid_thw[1] // spatial_merge_size
        width = curr_video_grid_thw[2] // spatial_merge_size
        video_token_indices = np.arange(curr_video_grid_thw[0]).reshape(-1, 1, 1)
        video_token_indices = np.broadcast_to(
            video_token_indices, (video_token_indices.shape[0], height, width)
        ).reshape(-1)
        video_token_indices = (
            (video_token_indices + shift)
            * next(iter([video_second_per_grid_t]))
            * position_id_per_seconds
        )
        video_data_index, audio_data_index = 0, 0
        updates = [audio_start_token_id]
        while video_data_index < len(video_token_indices) and audio_data_index < len(
            audio_token_indices
        ):
            if (
                video_token_indices[video_data_index]
                <= audio_token_indices[audio_data_index]
            ):
                updates += [video_token_id]
                video_data_index += 1
            else:
                updates += [audio_token_id]
                audio_data_index += 1
        if video_data_index < len(video_token_indices):
            updates += [video_token_id] * (len(video_token_indices) - video_data_index)
        if audio_data_index < len(audio_token_indices):
            updates += [audio_token_id] * (len(audio_token_indices) - audio_data_index)
        updates += [audio_end_token_id]
        return updates

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        vocab = tokenizer.get_vocab()

        audio_token = processor.audio_token
        image_token = processor.image_token
        video_token = processor.video_token
        audio_token_id = vocab[audio_token]
        image_token_id = vocab[image_token]
        video_token_id = vocab[video_token]

        out_mm_data = out_mm_kwargs.get_data()
        audio_feature_lengths = out_mm_data.get("audio_feature_lengths")
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        if audio_feature_lengths is None and feature_attention_mask is None:
            audio_output_lengths = []
        elif audio_feature_lengths is not None:
            _, audio_output_lens = _get_feat_extract_output_lengths(
                audio_feature_lengths
            )
            audio_output_lengths = audio_output_lens.tolist()
        elif feature_attention_mask is not None:
            assert isinstance(feature_attention_mask, ms.Tensor)
            _, audio_output_lens = _get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )
            audio_output_lengths = audio_output_lens.tolist()

        # number of audios read from video.
        audio_in_video_item_idx = 0
        audio_item_idx = 0

        def get_replacement_qwen2_audio(item_idx: int):
            nonlocal audio_item_idx
            item_idx += audio_in_video_item_idx

            audio_item_idx += 1

            num_features = audio_output_lengths[item_idx]
            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio = audios.get(item_idx)
                raise ValueError(
                    f"The audio {audio} (len={len(audio)}) is too short "
                    "to be represented inside the model"
                )

            return [audio_token_id] * num_features

        def get_replacement_qwen2_vision(item_idx: int, modality: str):
            grid_thw = out_mm_data[f"{modality}_grid_thw"][item_idx]
            assert isinstance(grid_thw, ms.Tensor)
            merge_length = image_processor.merge_size**2

            token_id = image_token_id if modality == "image" else video_token_id
            return [token_id] * (int(grid_thw.prod()) // merge_length)

        use_audio_in_video = hf_processor_mm_kwargs.get("use_audio_in_video", False)
        thinker_config = self.info.get_hf_config()

        def get_replacement_qwen2_use_audio_in_video(item_idx: int):
            nonlocal audio_in_video_item_idx
            audio_num_features = audio_output_lengths[audio_item_idx + item_idx]
            video_grid_thw = out_mm_data["video_grid_thw"][item_idx]

            audio_in_video_item_idx += 1

            second_per_grid_ts = hf_processor_mm_kwargs.get("second_per_grid_ts", None)
            if second_per_grid_ts:
                video_second_per_grid_t = second_per_grid_ts[item_idx]
            else:
                video_second_per_grid_t = 1.0

            return self.get_updates_use_audio_in_video(
                thinker_config=thinker_config,
                audio_len=audio_num_features,
                video_grid_thw=video_grid_thw,
                video_second_per_grid_t=video_second_per_grid_t,
            )

        video_replacement_fn = (
            get_replacement_qwen2_use_audio_in_video
            if use_audio_in_video
            else partial(get_replacement_qwen2_vision, modality="video")
        )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_qwen2_audio,
            ),
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=partial(get_replacement_qwen2_vision, modality="image"),
            ),
            PromptReplacement(
                modality="video",
                target=video_token,
                replacement=video_replacement_fn,
            ),
        ]

    def _validate_mm_placeholders(
        self,
        mm_placeholders: Mapping[str, list[PlaceholderFeaturesInfo]],
        mm_item_counts: Mapping[str, int],
    ) -> None:
        BaseMultiModalProcessor[
            Qwen2_5OmniThinkerProcessingInfo
        ]._validate_mm_placeholders(self, mm_placeholders, mm_item_counts)

    def _get_raw_input_ids(
        self,
        token_ids: list[int],
        use_audio_in_video: bool = False,
    ) -> list[int]:
        tokenizer = self.info.get_tokenizer()
        vision_bos_token = tokenizer.encode(tokenizer.vision_bos_token)[0]
        vision_eos_token = tokenizer.encode(tokenizer.vision_eos_token)[0]
        audio_bos_token = tokenizer.encode(tokenizer.audio_bos_token)[0]
        audio_eos_token = tokenizer.encode(tokenizer.audio_eos_token)[0]
        audio_token = tokenizer.encode("<|audio_pad|>")[0]
        image_token = tokenizer.encode("<|image_pad|>")[0]
        video_token = tokenizer.encode("<|video_pad|>")[0]

        result = token_ids[:]
        if use_audio_in_video:
            while True:
                start = None
                for i in range(len(result) - 1):
                    if result[i : i + 2] == [vision_bos_token, audio_bos_token]:
                        start = i
                        break
                if start is not None:
                    end = None
                    for i in range(start + 2, len(result) - 1):
                        if result[i : i + 2] == [audio_eos_token, vision_eos_token]:
                            end = i
                            break
                    if end is not None:
                        result = (
                            result[:start]
                            + [vision_bos_token, video_token, vision_eos_token]
                            + result[end + 2 :]
                        )
                else:
                    break

        for mm_token in [audio_token, image_token, video_token]:
            compressed = []
            for x in result:
                if x != mm_token or (not compressed or compressed[-1] != mm_token):
                    compressed.append(x)
            result = compressed

        return result


class Qwen3OmniMoeConditionalGenerationMixin(Qwen2_5OmniConditionalGenerationMixin):
    def _validate_and_reshape_mm_tensor(
        self, mm_input: object, name: str, dim: int = 0
    ) -> ms.Tensor:
        if not isinstance(mm_input, (ms.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")
        if name == "feature_attention_mask":
            dim = -1
        if isinstance(mm_input, ms.Tensor):
            return mint.concat(list(mm_input), dim=dim)
        else:
            if isinstance(mm_input[0], list):
                return mint.concat(
                    [mint.concat(mm_input[i], dim=dim) for i in range(len(mm_input))],
                    dim=dim,
                )
            else:
                return mint.concat(mm_input, dim=dim)

    def _process_audio_input(
        self,
        audio_input: Qwen2AudioInputs,
        audio_hashes: list[str] = None,
        cached_audio_features: ms.Tensor = None,
    ) -> ms.Tensor:
        input_features = audio_input["input_features"]
        audio_feature_lengths = audio_input["audio_feature_lengths"]

        if input_features.ndim == 3:
            assert input_features.shape[0] == 1
            input_features = input_features.squeeze(0)

        if not isinstance(audio_feature_lengths, ms.Tensor):
            audio_feature_lengths = mint.cat(audio_feature_lengths)
        if audio_feature_lengths.ndim == 2:
            audio_feature_lengths = audio_feature_lengths.reshape(-1)

        audio_feat_lengths, audio_output_lengths = _get_feat_extract_output_lengths(
            audio_feature_lengths
        )

        audio_outputs = self.audio_tower(
            input_features.astype(self.audio_tower.dtype),
            feature_lens=audio_feature_lengths,
            aftercnn_lens=audio_feat_lengths,
        )
        audio_features = audio_outputs.last_hidden_state
        return audio_features.split(audio_output_lengths.tolist())

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


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniMoeThinkerForConditionalGeneration(
    NativeModel,
    SupportsMultiModal,
    Qwen3OmniMoeConditionalGenerationMixin,
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "thinker.lm_head.": "language_model.lm_head.",
            "thinker.model.": "language_model.model.",
            "thinker.": "",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        if modality.startswith("audio"):
            return "<|audio_start|><|audio_pad|><|audio_end|>"

        raise ValueError("Only image, video or audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        thinker_config: Qwen3OmniMoeThinkerConfig = (
            vllm_config.model_config.hf_config.thinker_config
        )
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = thinker_config
        self.multimodal_config = multimodal_config

        # self.audio_tower = Qwen3OmniMoeAudioEncoder(thinker_config.audio_config)

        # attn_backend_override = (
        #     multimodal_config.mm_encoder_attn_backend
        #     if multimodal_config is not None
        #     else None
        # )
        self.visual = Qwen3Omni_VisionTransformer(
            vision_config=thinker_config.vision_config,
            norm_eps=getattr(thinker_config.text_config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )
        self.visual.set_model_inputs()
        self.visual.construct = ms.jit(function=self.visual, jit_level='O0')

        self.quant_config = quant_config

        self.language_model = Qwen3MoeLLMForCausalLM(
            vllm_config=vllm_config.with_hf_config(
                thinker_config.text_config, architectures=["Qwen3MoeForCausalLM"]
            ),
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.model = self.language_model.model
        self.lm_head = self.language_model.lm_head
        self.common_preprocess(vllm_config, prefix)

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        self.use_deepstack = hasattr(
            thinker_config.vision_config, "deepstack_visual_indexes"
        )
        self.deepstack_num_level = (
            len(thinker_config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        # register buffer for deepstack
        self.deepstack_input_embeds = (
            [
                mint.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    thinker_config.text_config.hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]
            if self.use_deepstack
            else None
        )
        self.visual_dim = thinker_config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

    def common_preprocess(self, vllm_config, prefix=""):
        self.set_modules({
            "thinker.visual.model": self.visual,
            "thinker.model": self.language_model.model,
            "thinker.lm_head": self.language_model.lm_head
        })
        self.casual_mask = MultiModalLowerTriangularMask(
            dtype=self.model_config.dtype,
            max_model_len=self.model_config.max_model_len)
        self.kv_caches = [
            AttentionWrapper() for i in range(self.config.num_hidden_layers)
        ]

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(self.config.num_hidden_layers):
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
                    num_tokens,
                    self.config.text_config.hidden_size,
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
            if (
                input_key in ("input_audio_features")
                and "audio" not in mm_input_by_modality
            ):
                mm_input_by_modality["audio"] = self._parse_and_validate_audio_input(
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
            if modality == "audio":
                audio_embeddings = self._process_audio_input(multimodal_input)
                multimodal_embeddings += tuple(audio_embeddings)
        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: ms.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: ms.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> ms.Tensor:
        inputs_embeds = self._get_text_embeddings(
            input_ids,
            self.language_model.get_input_embeddings,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

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

            deepstack_input_embeds = mint.zeros_like(
                inputs_embeds.view(inputs_embeds.shape[0], multiscale_len * inputs_embeds.shape[1])
            )
            deepstack_input_embeds = merge_multimodal_embeddings(
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

        inputs_embeds = merge_multimodal_embeddings(
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
        for name, weight in weights:
            if "visual." in name:
                self.visual.load_weights([(name, weight)], params_dict)
            elif "language_model." in name:
                self.language_model.load_weights([(name, weight)], params_dict)
            else:
                # Handle other weights
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, weight)
        return set()

    # def get_mrope_input_positions(
    #     self,
    #     input_tokens: list[int],
    #     hf_config: PretrainedConfig,
    #     image_grid_thw: list[list[int]] | ms.Tensor | None,
    #     video_grid_thw: list[list[int]] | ms.Tensor | None,
    #     second_per_grid_ts: list[float] | None = None,
    #     context_len: int = 0,
    #     seq_len: int | None = None,
    #     audio_feature_lengths: ms.Tensor | None = None,
    #     use_audio_in_video: bool = False,
    # ) -> tuple[ms.Tensor, int]:
    #     config = hf_config.thinker_config
    #     if isinstance(image_grid_thw, list):
    #         image_grid_thw = ms.Tensor(image_grid_thw)
    #     if isinstance(video_grid_thw, list):
    #         video_grid_thw = ms.Tensor(video_grid_thw)
    #     input_ids = ms.Tensor(input_tokens)
    #     if input_ids is None or input_ids.ndim != 1:
    #         raise ValueError("_omni3_get_input_positions_tensor expects 1D input_ids")

    #     seq_len = input_ids.shape[0]
    #     if audio_feature_lengths is not None and not isinstance(
    #         audio_feature_lengths, ms.Tensor
    #     ):
    #         audio_feature_lengths = ms.Tensor(
    #             audio_feature_lengths, dtype=ms.int64
    #         )
    #     if second_per_grid_ts is None:
    #         if video_grid_thw is not None and video_grid_thw.numel() > 0:
    #             second_per_grids = mint.ones(
    #                 video_grid_thw.shape[0], dtype=ms.float32
    #             )
    #         else:
    #             second_per_grids = ms.Tensor([], dtype=ms.float32)
    #     else:
    #         second_per_grids = ms.Tensor(second_per_grid_ts, dtype=ms.float32)

    #     spatial_merge_size = config.vision_config.spatial_merge_size
    #     image_token_id = config.image_token_id
    #     video_token_id = config.video_token_id
    #     audio_token_id = config.audio_token_id
    #     vision_start_token_id = config.vision_start_token_id
    #     audio_start_token_id = config.audio_start_token_id
    #     position_id_per_seconds = config.position_id_per_seconds

    #     vision_start_indices = mint.argwhere(
    #         input_ids == vision_start_token_id
    #     ).squeeze(1)
    #     if vision_start_indices.numel() > 0:
    #         vision_tokens = input_ids[vision_start_indices + 1]
    #     else:
    #         vision_tokens = mint.empty((0,), dtype=input_ids.dtype)
    #     audio_nums = mint.sum(input_ids == audio_start_token_id)
    #     image_nums = (vision_tokens == image_token_id).sum()
    #     video_nums = (
    #         (vision_tokens == audio_start_token_id).sum()
    #         if use_audio_in_video
    #         else (vision_tokens == video_token_id).sum()
    #     )

    #     llm_pos_ids_list: list[ms.Tensor] = []
    #     st = 0
    #     image_idx = 0
    #     video_idx = 0
    #     audio_idx = 0
    #     remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums  # noqa: E501
    #     multimodal_nums = (
    #         image_nums + audio_nums
    #         if use_audio_in_video
    #         else image_nums + video_nums + audio_nums
    #     )  # noqa: E501

    #     for _ in range(multimodal_nums):
    #         st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #         if (image_token_id in input_tokens or video_token_id in input_tokens) and (
    #             remain_videos > 0 or remain_images > 0
    #         ):
    #             ed_vision_start = input_tokens.index(vision_start_token_id, st)
    #         else:
    #             ed_vision_start = len(input_tokens) + 1
    #         if audio_token_id in input_tokens and remain_audios > 0:
    #             ed_audio_start = input_tokens.index(audio_start_token_id, st)
    #         else:
    #             ed_audio_start = len(input_tokens) + 1
    #         min_ed = min(ed_vision_start, ed_audio_start)

    #         if min_ed == ed_audio_start:
    #             text_len = min_ed - st
    #             if text_len != 0:
    #                 st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #                 llm_pos_ids_list.append(
    #                     mint.arange(text_len, dtype=ms.int64)
    #                     .view(1, -1)
    #                     .expand(3, -1)
    #                     + st_idx
    #                 )
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             bos_len = 1
    #             llm_pos_ids_list.append(
    #                 mint.arange(bos_len, dtype=ms.int64).view(1, -1).expand(3, -1)
    #                 + st_idx
    #             )
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             _, audio_len = _get_feat_extract_output_lengths(
    #                 audio_feature_lengths[audio_idx]
    #             )
    #             llm_pos_ids = (
    #                 mint.arange(audio_len, dtype=ms.int64).view(1, -1).expand(3, -1)
    #                 + st_idx
    #             )
    #             llm_pos_ids_list.append(llm_pos_ids)
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             eos_len = 1
    #             llm_pos_ids_list.append(
    #                 mint.arange(eos_len, dtype=ms.int64).view(1, -1).expand(3, -1)
    #                 + st_idx
    #             )
    #             st += text_len + bos_len + audio_len + eos_len
    #             audio_idx += 1
    #             remain_audios -= 1
    #         elif (
    #             min_ed == ed_vision_start
    #             and input_ids[ed_vision_start + 1] == image_token_id
    #         ):
    #             text_len = min_ed - st
    #             if text_len != 0:
    #                 st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #                 llm_pos_ids_list.append(
    #                     mint.arange(text_len, dtype=ms.int64)
    #                     .view(1, -1)
    #                     .expand(3, -1)
    #                     + st_idx
    #                 )
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             bos_len = 1
    #             llm_pos_ids_list.append(
    #                 mint.arange(bos_len, dtype=ms.int64).view(1, -1).expand(3, -1)
    #                 + st_idx
    #             )
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             grid_t = image_grid_thw[image_idx][0]
    #             grid_hs = image_grid_thw[:, 1]
    #             grid_ws = image_grid_thw[:, 2]
    #             t_index = mint.arange(grid_t) * position_id_per_seconds
    #             llm_pos_ids = get_llm_pos_ids_for_vision(
    #                 st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
    #             )
    #             image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
    #             llm_pos_ids_list.append(llm_pos_ids)
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             eos_len = 1
    #             llm_pos_ids_list.append(
    #                 mint.arange(eos_len, dtype=ms.int64).view(1, -1).expand(3, -1)
    #                 + st_idx
    #             )
    #             st += text_len + bos_len + image_len + eos_len
    #             image_idx += 1
    #             remain_images -= 1
    #         elif (
    #             min_ed == ed_vision_start
    #             and input_ids[ed_vision_start + 1] == video_token_id
    #             and not use_audio_in_video
    #         ):
    #             text_len = min_ed - st
    #             if text_len != 0:
    #                 st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #                 llm_pos_ids_list.append(
    #                     mint.arange(text_len, dtype=ms.int64)
    #                     .view(1, -1)
    #                     .expand(3, -1)
    #                     + st_idx
    #                 )
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             bos_len = 1
    #             llm_pos_ids_list.append(
    #                 mint.arange(bos_len, dtype=ms.int64).view(1, -1).expand(3, -1)
    #                 + st_idx
    #             )
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             grid_t = video_grid_thw[video_idx][0]
    #             grid_hs = video_grid_thw[:, 1]
    #             grid_ws = video_grid_thw[:, 2]
    #             t_index = (
    #                 mint.arange(grid_t)
    #                 * float(second_per_grids[video_idx].item())
    #                 * position_id_per_seconds
    #             )
    #             llm_pos_ids = get_llm_pos_ids_for_vision(
    #                 st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
    #             )
    #             video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
    #             llm_pos_ids_list.append(llm_pos_ids)
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             eos_len = 1
    #             llm_pos_ids_list.append(
    #                 mint.arange(eos_len, dtype=ms.int64).view(1, -1).expand(3, -1)
    #                 + st_idx
    #             )
    #             st += text_len + bos_len + video_len + eos_len
    #             video_idx += 1
    #             remain_videos -= 1
    #         elif (
    #             min_ed == ed_vision_start
    #             and ed_vision_start + 1 == ed_audio_start
    #             and use_audio_in_video
    #         ):
    #             text_len = min_ed - st
    #             if text_len != 0:
    #                 st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #                 llm_pos_ids_list.append(
    #                     mint.arange(text_len, dtype=ms.int64)
    #                     .view(1, -1)
    #                     .expand(3, -1)
    #                     + st_idx
    #                 )
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             bos_len = 1
    #             bos_block = (
    #                 mint.arange(bos_len, dtype=ms.int64).view(1, -1).expand(3, -1)
    #                 + st_idx
    #             )
    #             llm_pos_ids_list.append(bos_block)
    #             llm_pos_ids_list.append(bos_block)
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             _, audio_len = _get_feat_extract_output_lengths(
    #                 audio_feature_lengths[audio_idx]
    #             )
    #             audio_llm_pos_ids = (
    #                 mint.arange(audio_len, dtype=ms.int64).view(1, -1).expand(3, -1)
    #                 + st_idx
    #             )
    #             grid_t = video_grid_thw[video_idx][0]
    #             grid_hs = video_grid_thw[:, 1]
    #             grid_ws = video_grid_thw[:, 2]
    #             t_index = (
    #                 mint.arange(grid_t)
    #                 * float(second_per_grids[video_idx].item())
    #                 * position_id_per_seconds
    #             )
    #             video_llm_pos_ids = get_llm_pos_ids_for_vision(
    #                 st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
    #             )
    #             video_data_index, audio_data_index = 0, 0
    #             while (
    #                 video_data_index < video_llm_pos_ids.shape[-1]
    #                 and audio_data_index < audio_llm_pos_ids.shape[-1]
    #             ):
    #                 if (
    #                     video_llm_pos_ids[0][video_data_index]
    #                     <= audio_llm_pos_ids[0][audio_data_index]
    #                 ):
    #                     llm_pos_ids_list.append(
    #                         video_llm_pos_ids[
    #                             :, video_data_index : video_data_index + 1
    #                         ]
    #                     )
    #                     video_data_index += 1
    #                 else:
    #                     llm_pos_ids_list.append(
    #                         audio_llm_pos_ids[
    #                             :, audio_data_index : audio_data_index + 1
    #                         ]
    #                     )
    #                     audio_data_index += 1
    #             if video_data_index < video_llm_pos_ids.shape[-1]:
    #                 llm_pos_ids_list.append(
    #                     video_llm_pos_ids[
    #                         :, video_data_index : video_llm_pos_ids.shape[-1]
    #                     ]
    #                 )
    #             if audio_data_index < audio_llm_pos_ids.shape[-1]:
    #                 llm_pos_ids_list.append(
    #                     audio_llm_pos_ids[
    #                         :, audio_data_index : audio_llm_pos_ids.shape[-1]
    #                     ]
    #                 )
    #             video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             eos_len = 1
    #             eos_block = (
    #                 mint.arange(eos_len, dtype=ms.int64).view(1, -1).expand(3, -1)
    #                 + st_idx
    #             )
    #             llm_pos_ids_list.append(eos_block)
    #             llm_pos_ids_list.append(eos_block)
    #             st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2  # noqa: E501
    #             audio_idx += 1
    #             video_idx += 1
    #             remain_videos -= 1
    #             remain_audios -= 1

    #     if st < len(input_tokens):
    #         st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #         text_len = len(input_tokens) - st
    #         llm_pos_ids_list.append(
    #             mint.arange(text_len, dtype=ms.int64).view(1, -1).expand(3, -1)
    #             + st_idx
    #         )

    #     llm_positions = mint.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    #     if llm_positions.shape[1] != seq_len:
    #         raise RuntimeError("Position ids length mismatch with input ids length")

    #     mrope_position_delta = llm_positions.max() + 1 - seq_len
    #     return llm_positions, mrope_position_delta

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
        rotary_pos_emb_full = self.rotary_pos_emb_total(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> ms.Tensor:
        num_grid_per_side = self.num_grid_per_side
        m_size = self.spatial_merge_size
        hidden_dim = self.pos_embed.embedding_dim

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
            weights = weights.astype(self.dtype)

            embeds = self.pos_embed(indices)
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
