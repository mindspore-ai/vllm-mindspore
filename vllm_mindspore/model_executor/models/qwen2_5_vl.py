# SPDX-License-Identifier: Apache-2.0
# type: ignore
# isort:skip_file
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2_5_vl.py
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
# ============================================================================
"""Inference-only Qwen2.5-VL model compatible with HuggingFace weights."""
import os
from functools import partial
from typing import Callable, Iterable, Mapping, Optional, Set, Tuple, Union, Dict, Any

import math
import mindspore as ms
import mindspore.nn as nn
import mindspore.mint as mint
import mindspore.ops as ops
import mindspore.mint.nn.functional as F
from mindspore import dtype as mstype

from vllm_mindspore.model_executor.layers.layernorm import RMSNorm
from vllm_mindspore.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm_mindspore.model_executor.layers.logits_processor import LogitsProcessor
from vllm_mindspore.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm_mindspore.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm_mindspore.model_executor.model_loader.weight_utils import default_weight_loader
from vllm_mindspore.model_executor.models.model_base import NativeModel, AttentionWrapper
from vllm_mindspore.model_executor.models.interfaces import SupportsMultiModal
from vllm_mindspore.model_executor.models.qwen2 import Qwen2Model  # type: ignore[attr-defined]
from vllm_mindspore.model_executor.models.utils import PPMissingLayer, WeightsMapper, maybe_prefix, \
    merge_multimodal_embeddings
from vllm_mindspore.model_executor.models.attention_mask import MultiModalLowerTriangularMask
from vllm_mindspore.distributed.communication_op import AllGatherFromModelParallelRegion

from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen2_vl import Qwen2VLDummyInputsBuilder as Qwen2_5_VLDummyInputsBuilder
from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalProcessor
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLImageInputs, Qwen2_5_VLVideoInputs, \
    Qwen2_5_VLImagePixelInputs, Qwen2_5_VLImageEmbeddingInputs, Qwen2_5_VLVideoPixelInputs, \
    Qwen2_5_VLVideoEmbeddingInputs, Qwen2_5_VLProcessingInfo

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargs
from vllm.multimodal.processing import PromptReplacement
from vllm.multimodal.parse import MultiModalDataItems
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from vllm.distributed import utils as dist_utils
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import uses_mrope

logger = init_logger(__name__)

_ACTIVATION_REGISTRY = {"silu": F.silu}

# === Vision Inputs === #


class _Qwen2VLMultiModalProcessor(Qwen2VLMultiModalProcessor):

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


class Qwen2_5_VisionMLP(nn.Cell):

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 bias: bool = False,
                 act_fn: Callable[[ms.Tensor], ms.Tensor] = F.silu,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(in_features,
                                              hidden_features,
                                              bias=bias,
                                              quant_config=quant_config,
                                              prefix=f"{prefix}.gate_proj",
                                              params_dtype=ms.bfloat16)
        self.up_proj = ColumnParallelLinear(in_features,
                                            hidden_features,
                                            bias=bias,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.up_proj",
                                            params_dtype=ms.bfloat16)
        self.down_proj = RowParallelLinear(hidden_features,
                                           in_features,
                                           bias=bias,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.down_proj",
                                           params_dtype=ms.bfloat16)
        self.act_fn = act_fn

    def construct(self, x: ms.Tensor):
        x_gate, _ = self.gate_proj(x)
        x_gate = self.act_fn(x_gate)
        x_up, _ = self.up_proj(x)
        x_down, _ = self.down_proj(x_gate * x_up)
        return x_down


def apply_rotary_pos_emb_flashatt(
        q: ms.Tensor, k: ms.Tensor, cos: ms.Tensor,
        sin: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
    q_embed = ops.rotary_position_embedding(q.float(), cos, sin).type_as(q)
    k_embed = ops.rotary_position_embedding(k.float(), cos, sin).type_as(k)
    return q_embed, k_embed


class Qwen2_5_VisionAttention(nn.Cell):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Per attention head and per partition values.
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads)
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, self.tp_size)
        self.num_heads = num_heads
        self.head_dim = self.hidden_size_per_attention_head

        self.qkv = ColumnParallelLinear(input_size=embed_dim,
                                        output_size=3 * projection_size,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.qkv",
                                        params_dtype=ms.bfloat16)
        self.proj = RowParallelLinear(input_size=projection_size,
                                      output_size=embed_dim,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.proj",
                                      params_dtype=ms.bfloat16)
        self.tensor_model_parallel_all_gather = AllGatherFromModelParallelRegion(
        )

    def split_tensor_along_last_dim(
        self,
        tensor: ms.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
    ):
        """ Split a tensor along its last dimension.

            Arguments:
                tensor: input tensor.
                num_partitions: number of partitions to split the tensor
                contiguous_split_chunks: If True, make each chunk contiguous
                                         in memory.

            Returns:
                A list of Tensors
        """
        # Get the size and dimension.
        last_dim = tensor.dim() - 1
        last_dim_size = dist_utils.divide(tensor.shape[last_dim],
                                          num_partitions)
        # Split.
        tensor_list = mint.split(tensor, last_dim_size, dim=last_dim)
        # NOTE: torch.split does not create contiguous tensors by default.

        return tensor_list

    def split_qkv(self, qkv: ms.Tensor) -> tuple[ms.Tensor, ...]:
        # [s, 3 * head * head_dim]
        seq_len, _ = qkv.shape
        if self.tp_size > 1:
            qkv = self.tensor_model_parallel_all_gather(qkv)

        # [s, 3 * head * head_dim] -> 3 * [s, head * head_dim]
        q, k, v = mint.chunk(qkv, 3, dim=-1)

        # 3 * [s, head * head_dim]
        if self.tp_size > 1:
            splitter = partial(self.split_tensor_along_last_dim,
                               num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
            v = splitter(v)[self.tp_rank]

        # 3 * [s, head * head_dim] -> 3 * [s, head, head_dim]
        new_shape = (seq_len, self.num_attention_heads_per_partition,
                     self.hidden_size_per_attention_head)
        q, k, v = (x.view(*new_shape) for x in (q, k, v))
        return q, k, v

    def construct(
        self,
        x: ms.Tensor,
        cu_seqlens: ms.Tensor,
        position_embeddings: Tuple[ms.Tensor, ms.Tensor],
    ) -> ms.Tensor:
        seq_length = x.shape[0]
        x, _ = self.qkv(x)
        q, k, v = self.split_qkv(x)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_flashatt(mint.unsqueeze(q, 0),
                                             mint.unsqueeze(k, 0), cos, sin)

        q = mint.squeeze(q, 0)
        k = mint.squeeze(k, 0)

        context_layer = ops.flash_attention_score(
            q,
            k,
            v,
            self.num_heads // self.tp_size,
            actual_seq_qlen=cu_seqlens,
            actual_seq_kvlen=cu_seqlens,
            scalar_value=1 / math.sqrt(q.shape[-1]),
            input_layout="TND",
        ).reshape(seq_length, -1)
        output, _ = self.proj(context_layer)
        return output


class Qwen2_5_VisionBlock(nn.Cell):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[ms.Tensor], ms.Tensor] = F.silu,
        norm_layer: Optional[Callable[[int], nn.Cell]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(mint.nn.LayerNorm,
                                 eps=1e-6,
                                 dtype=ms.bfloat16)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Qwen2_5_VisionAttention(embed_dim=dim,
                                            num_heads=num_heads,
                                            projection_size=dim,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.attn")
        self.mlp = Qwen2_5_VisionMLP(dim,
                                     mlp_hidden_dim,
                                     act_fn=act_fn,
                                     bias=True,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.mlp")

    def construct(
            self, x: ms.Tensor, cu_seqlens: ms.Tensor,
            position_embeddings: Tuple[ms.Tensor, ms.Tensor]) -> ms.Tensor:
        x = x + self.attn(self.norm1(x),
                          cu_seqlens=cu_seqlens,
                          position_embeddings=position_embeddings)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen2_5_VisionPatchEmbed(nn.Cell):

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

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = mint.nn.Conv3d(in_channels,
                                   hidden_size,
                                   kernel_size=kernel_size,
                                   stride=kernel_size,
                                   bias=False,
                                   dtype=ms.bfloat16)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        L, C = x.shape
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size,
                   self.patch_size)
        x = self.proj(x).view(L, self.hidden_size)
        return x


class Qwen2_5_VisionPatchMerger(nn.Cell):

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Optional[Callable[[int], nn.Cell]] = None,
        spatial_merge_size: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        if norm_layer is None:
            norm_layer = partial(mint.nn.LayerNorm,
                                 eps=1e-6,
                                 dtype=ms.bfloat16)
        self.ln_q = norm_layer(context_dim)
        self.mlp = nn.CellList([
            ColumnParallelLinear(self.hidden_size,
                                 self.hidden_size,
                                 bias=True,
                                 quant_config=quant_config,
                                 prefix=f"{prefix}.mlp.0",
                                 params_dtype=ms.bfloat16),
            nn.GELU(),
            RowParallelLinear(self.hidden_size,
                              d_model,
                              bias=True,
                              quant_config=quant_config,
                              prefix=f"{prefix}.mlp.2",
                              params_dtype=ms.bfloat16),
        ])

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.ln_q(x)
        x = x.view(-1, self.hidden_size)

        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x_parallel, _ = mlp_fc1(x)
        x_parallel = mlp_act(x_parallel)
        out, _ = mlp_fc2(x_parallel)
        return out


class Qwen2_5_VisionRotaryEmbedding(nn.Cell):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.inv_freq = 1.0 / (theta**(
            mint.arange(0, dim, 2, dtype=ms.float32) / dim))
        self._seq_len_cached = 0
        self._freqs_cached = None

    def update_freqs_cache(self, seqlen: int) -> None:
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            self.inv_freq = 1.0 / (self.theta**(
                mint.arange(0, self.dim, 2, dtype=ms.float32) / self.dim))
            seq = mint.arange(seqlen, dtype=self.inv_freq.dtype)
            freqs = mint.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def construct(self, seqlen: int) -> ms.Tensor:
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]  # type: ignore[index]


class Qwen2_5_VisionTransformer(nn.Cell):

    def __init__(
        self,
        vision_config,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        depth = vision_config.depth
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads

        # args for get_window_index
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.spatial_merge_unit = self.spatial_merge_size**2

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size,
        )

        norm_layer = partial(RMSNorm, eps=norm_eps, params_dtype=ms.bfloat16)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.CellList([
            Qwen2_5_VisionBlock(
                dim=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_dim=vision_config.intermediate_size,
                act_fn=_ACTIVATION_REGISTRY[vision_config.hidden_act],
                norm_layer=norm_layer,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{layer_idx}")
            for layer_idx in range(depth)
        ])
        self.merger = Qwen2_5_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            norm_layer=norm_layer,
            spatial_merge_size=self.spatial_merge_size,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
        )
        from mindspore.communication.management import get_rank
        self.rank_id = get_rank()

    def set_model_inputs(self):
        dyn_x = ms.Tensor(shape=[None, None], dtype=self.dtype)
        dyn_rotary_pos_emb = ms.Tensor(shape=[None, None],
                                       dtype=mstype.float32)
        dyn_window_index = ms.Tensor(shape=[None], dtype=mstype.int64)
        dyn_cu_window_seqlens = ms.Tensor(shape=[None], dtype=mstype.int64)
        dyn_grid_thw = ms.Tensor(shape=[None, None], dtype=mstype.int64)

        self.set_inputs(
            dyn_x,
            dyn_rotary_pos_emb,
            dyn_window_index,
            dyn_cu_window_seqlens,
            dyn_grid_thw,
        )

    @property
    def dtype(self) -> ms.Type:
        return self.patch_embed.proj.weight.dtype

    def construct(
        self,
        x: ms.Tensor,
        rotary_pos_emb: ms.Tensor,
        window_index: ms.Tensor,
        cu_window_seqlens: ms.Tensor,
        grid_thw: ms.Tensor,
    ) -> ms.Tensor:
        hidden_states = x.to(dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        cu_window_seqlens = cu_window_seqlens.astype(ms.int32)
        cu_window_seqlens = mint.unique_consecutive(cu_window_seqlens)
        seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index]
        rotary_pos_emb = rotary_pos_emb.reshape(1, seq_len, 1, -1)
        emb = mint.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (mint.cos(emb), mint.sin(emb))

        grid_thw_1 = grid_thw.index_select(1, ms.Tensor([1])).reshape(-1)
        grid_thw_2 = grid_thw.index_select(1, ms.Tensor([2])).reshape(-1)
        grid_thw_0 = grid_thw.index_select(1, ms.Tensor([0])).reshape(-1)
        cu_seqlens = mint.cumsum(mint.repeat_interleave(
            grid_thw_1 * grid_thw_2, grid_thw_0),
                                 dim=0,
                                 dtype=ms.int32)

        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)
        # transformers
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            hidden_states = blk(hidden_states,
                                cu_seqlens=cu_seqlens_now,
                                position_embeddings=position_embeddings)

        # adapter
        hidden_states = self.merger(hidden_states)
        reverse_indices = mint.argsort(window_index)
        hidden_states = hidden_states[reverse_indices]
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, ms.Tensor]],
                     params_dict: Dict[str, ms.Parameter]) -> Set[str]:
        loaded_params: Set[str] = set()
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

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
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


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


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder)
class Qwen2_5_VLForConditionalGeneration(NativeModel, SupportsMultiModal):
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

        self.visual = Qwen2_5_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=self._maybe_ignore_quant_config(quant_config),
            prefix=maybe_prefix(prefix, "visual"),
        )
        self.visual = ms.jit(
            function=self.visual,
            jit_level='O0') if self.is_graph_mode else self.visual

        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              params_dtype=ms.bfloat16,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
            self.logits_processor = LogitsProcessor(config.vocab_size)
            self.sampler = get_sampler()
        else:
            self.lm_head = PPMissingLayer()

        self.common_preprocess(vllm_config, prefix)
        self.spatial_merge_size = config.vision_config.spatial_merge_size

        self.window_size = config.vision_config.window_size
        self.patch_size = config.vision_config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.hidden_size = config.vision_config.hidden_size
        self.num_heads = config.vision_config.num_heads
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        if self.is_graph_mode:
            self.visual.set_model_inputs()

    def common_preprocess(self, vllm_config, prefix=""):
        self.set_modules({
            "visual": self.visual,
            "model": self.model,
            "lm_head": self.lm_head
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

    def _maybe_ignore_quant_config(self, quant_config: QuantizationConfig):
        # GPTQ configs do not have a list of ignored modules, however AutoGPTQ
        # seems to avoid vision encoder sections for some models.
        # if isinstance(quant_config, (GPTQConfig, GPTQMarlinConfig)):
        #     return None
        return quant_config

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

    def rot_pos_emb(self, grid_thw: ms.Tensor) -> ms.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            t, h, w = t.item(), h.item(), w.item()
            hpos_ids = mint.arange(h).unsqueeze(1).expand((-1, w))
            wpos_ids = mint.arange(w).unsqueeze(0).expand((h, -1))

            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            pos_ids.append(
                mint.tile(mint.stack([hpos_ids, wpos_ids], dim=-1), (t, 1)))
        pos_ids = mint.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max().item()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index = []
        cu_window_seqlens = [ms.Tensor([0])]
        window_index_id = 0
        vit_merger_window_size = (self.window_size //
                                  self.spatial_merge_size // self.patch_size)

        for grid_t, grid_h, grid_w in grid_thw:
            grid_t, grid_h, grid_w = grid_t.item(), grid_h.item(), grid_w.item(
            )
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            index = mint.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), 'constant', -100)
            index_padded = index_padded.reshape(grid_t, num_windows_h,
                                                vit_merger_window_size,
                                                num_windows_w,
                                                vit_merger_window_size)
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t, num_windows_h * num_windows_w, vit_merger_window_size,
                vit_merger_window_size)
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = mint.cumsum(
                seqlens,
                0) * self.spatial_merge_unit + cu_window_seqlens[-1][-1]
            cu_window_seqlens.append(cu_seqlens_tmp)
            window_index_id += grid_t * llm_grid_h * llm_grid_w
        window_index = mint.cat(window_index, dim=0)
        cu_window_seqlens = mint.cat(cu_window_seqlens, dim=0)
        return window_index, cu_window_seqlens

    def _process_image_input(
            self, image_input: Qwen2_5_VLImageInputs) -> tuple[ms.Tensor, ...]:

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            os.environ[
                "MS_DISABLE_INTERNAL_KERNELS_LIST"] = "FlashAttentionScore"
            # compute position embedding
            rotary_pos_emb = self.rot_pos_emb(grid_thw)
            # windows attention
            window_index, cu_window_seqlens = self.get_window_index(grid_thw)
            image_embeds = self.visual(pixel_values, rotary_pos_emb,
                                       window_index, cu_window_seqlens,
                                       grid_thw)
            os.environ["MS_DISABLE_INTERNAL_KERNELS_LIST"] = ""

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return image_embeds.split(sizes.tolist())

    def _process_video_input(
            self, video_input: Qwen2_5_VLVideoInputs) -> tuple[ms.Tensor, ...]:

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(
                self.visual.dtype)
            os.environ[
                "MS_DISABLE_INTERNAL_KERNELS_LIST"] = "FlashAttentionScore"
            rotary_pos_emb = self.rot_pos_emb(grid_thw)
            # windows attention
            window_index, cu_window_seqlens = self.get_window_index(grid_thw)
            video_embeds = self.visual(pixel_values_videos, rotary_pos_emb,
                                       window_index, cu_window_seqlens,
                                       grid_thw)
            os.environ["MS_DISABLE_INTERNAL_KERNELS_LIST"] = ""

        # Split concatenated embeddings for each video item.
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

    def get_multimodal_embeddings(self,
                                  **kwargs) -> Optional[tuple[ms.Tensor, ...]]:

        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return None

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[ms.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                vision_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += vision_embeddings
            if modality == "videos":
                video_input = modalities["videos"]
                video_embeddings = self._process_video_input(video_input)
                multimodal_embeddings += video_embeddings
        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: ms.Tensor,
        multimodal_embeddings: Optional[tuple[ms.Tensor, ...]] = None,
    ) -> ms.Tensor:
        # input_ids = input_ids.to(mstype.int64)
        inputs_embeds = self.model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.config.image_token_id, self.config.video_token_id])
        os.environ["MS_DISABLE_INTERNAL_KERNELS_LIST"] = ""
        return inputs_embeds

    def get_input_embeddings_v0(
        self,
        input_ids: ms.Tensor,
        image_input: Optional[tuple[ms.Tensor, ...]] = None,
        video_input: Optional[tuple[ms.Tensor, ...]] = None,
    ) -> ms.Tensor:
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

    def forward(
        self,
        input_ids: ms.Tensor,
        positions: ms.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        **kwargs: object,
    ) -> Union[ms.Tensor, IntermediateTensors]:
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
                    assert positions.ndim == 2 and positions.shape[0] == 3, (
                        "multimodal section rotary embedding requires "
                        f"(3, seq_len) positions, but got {positions.shape}")
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids,
                    image_input=image_input,
                    video_input=video_input)
                input_ids = None
        hidden_states = self.exec_model(input_ids, positions,
                                        intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: ms.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[ms.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(self, logits: ms.Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(
        self, weights: Iterable[Tuple[str, ms.Tensor]]
    ) -> None:  # type: ignore[override]
        params_dict = self.get_params_dict()
        for name, weight in weights:
            if "visual." in name:
                self.visual.load_weights([(name, weight)], params_dict)
            else:
                self.model.load_weights([(name, weight)], params_dict)

        return None

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.",
            tower_model="visual.merger.")
