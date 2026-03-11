from collections.abc import Iterable
from typing import Optional

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.mint as mint
import mindspore.mint.nn.functional as F

from transformers.models.idefics2.configuration_idefics2 import (
    Idefics2Config,
    Idefics2VisionConfig,
)
from mindspore.ops.operations.nn_ops import PromptFlashAttention,FlashAttentionScore


from vllm.attention.layer import MultiHeadAttention  

from vllm.distributed import get_tensor_model_parallel_world_size 

from vllm_mindspore.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm_mindspore.model_executor.model_loader.weight_utils import default_weight_loader



_ACTIVATION_REGISTRY = {"gelu_pytorch_tanh": nn.GELU(approximate=True),"silu": nn.SiLU,"gelu": nn.GELU()}

class Idefics2VisionEmbeddings(nn.Cell):
    """
    MindSpore版 Idefics2VisionEmbeddings，支持变分辨率图片。
    """

    def __init__(self, config: Idefics2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = mint.nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
            bias=True,
        )
        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.num_positions = self.num_patches
        self.position_embedding = mint.nn.Embedding(self.num_positions, self.embed_dim)

    def construct(
        self,
        pixel_values: ms.Tensor,
        patch_attention_mask: ms.Tensor,
        tgt_sizes: Optional[ms.Tensor] = None
    ) -> ms.Tensor:
        batch_size, _, max_im_h, max_im_w = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.astype(target_dtype))
        embeddings = mint.transpose(mint.flatten(patch_embeds, 2), 1, 2)  # (B, N, C)
        max_nb_patches_h = max_im_h // self.patch_size
        max_nb_patches_w = max_im_w // self.patch_size
        boundaries = mint.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        position_ids = mint.full((batch_size, max_nb_patches_h * max_nb_patches_w), 0, dtype=ms.int64)

        for batch_idx in range(batch_size):
            p_attn_mask = patch_attention_mask[batch_idx].astype(ms.int64)
            if tgt_sizes is not None:
                nb_patches_h = int(tgt_sizes[batch_idx][0].asnumpy().item())
                nb_patches_w = int(tgt_sizes[batch_idx][1].asnumpy().item())
            else:
                nb_patches_h = int(p_attn_mask[:, 0].sum().asnumpy().item())
                nb_patches_w = int(p_attn_mask[0].sum().asnumpy().item())
            fractional_coords_h = mint.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = mint.arange(0, 1 - 1e-6, 1 / nb_patches_w)
            
            bucket_coords_h = mint.searchsorted( boundaries, fractional_coords_h, right=True)
            bucket_coords_w = mint.searchsorted( boundaries, fractional_coords_w, right=True)
            
            pos_ids = mint.flatten((bucket_coords_h[:, None] * self.num_patches_per_side +
                       bucket_coords_w))
            position_ids[batch_idx][patch_attention_mask[batch_idx].view((-1))] = pos_ids
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


class Idefics2VisionAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper (MindSpore版)"""

    def __init__(
        self,
        config: Idefics2VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads}).")
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        tp_size = 1  
        self.num_heads_per_partition = self.num_heads // tp_size

        self.qkv_proj = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            self.num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.out_proj = RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        
        self.attn = FlashAttentionScore(
                head_num=self.num_heads_per_partition,
                scale_value=self.scale,
                )

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        target_dtype = next(self.qkv_proj.get_parameters()).dtype
        if hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.astype(target_dtype)
        qkv, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = mint.chunk(qkv, 3, dim=-1)
        
        q_input = query_states.contiguous()
        k_input = key_states.contiguous()
        v_input = value_states.contiguous()
        _, _, _, out = self.attn(
                q_input,
                k_input,
                v_input,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        attn_output, _ = self.out_proj(out)
        return attn_output


class Idefics2VisionMLP(nn.Cell):
    def __init__(
        self,
        config: Idefics2VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.config = config
        self.activation_fn = _ACTIVATION_REGISTRY[config.hidden_act]
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        target_dtype = next(self.fc1.get_parameters()).dtype
        if hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.astype(target_dtype)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class Idefics2EncoderLayer(nn.Cell):
    def __init__(
        self,
        config: Idefics2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Idefics2VisionAttention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            use_data_parallel=use_data_parallel
            )
        self.layer_norm1 = mint.nn.LayerNorm([self.embed_dim], eps=config.layer_norm_eps)
        self.mlp = Idefics2VisionMLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            use_data_parallel=use_data_parallel)
        self.layer_norm2 = mint.nn.LayerNorm([self.embed_dim], eps=config.layer_norm_eps)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class Idefics2Encoder(nn.Cell):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention
    layers. Each layer is a [`Idefics2EncoderLayer`].
    """

    def __init__(
        self,
        config: Idefics2Config,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()

        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.CellList([
            Idefics2EncoderLayer(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{layer_idx}",
                use_data_parallel=use_data_parallel
            )
            for layer_idx in range(num_hidden_layers)
        ])

    def construct(self, inputs_embeds: ms.Tensor) -> ms.Tensor:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states


class Idefics2VisionTransformer(nn.Cell):
    def __init__(
        self,
        config: Idefics2VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: bool = True,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()

        embed_dim = config.hidden_size
        self.config = config
        self.use_data_parallel = use_data_parallel
        self.embeddings = Idefics2VisionEmbeddings(config)
        self.encoder = Idefics2Encoder(
            config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.encoder",
            use_data_parallel=use_data_parallel
        )

        num_hidden_layers = config.num_hidden_layers
        if len(self.encoder.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.encoder.layers)} layers."
            )

        self.require_post_norm = require_post_norm
        self.post_layernorm = (
            mint.nn.LayerNorm([embed_dim], eps=config.layer_norm_eps)
            if require_post_norm else mint.nn.Identity()
        )

    def get_input_embeddings(self):
        return self.embeddings

    def construct(
        self,
        pixel_values,
        patch_attention_mask: Optional[ms.Tensor] = None,
        tgt_sizes: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        hidden_states = self.embeddings(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
            tgt_sizes=tgt_sizes,
        )
        
        encoder_outputs = self.encoder(hidden_states)
        last_hidden_state = self.post_layernorm(encoder_outputs)
        return last_hidden_state

    def load_weights(self, weights: Iterable[tuple[str, ms.Tensor]],
                     params_dict: dict[str, ms.Parameter]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        
        loaded_params = set()
        layer_count = len(self.encoder.layers)

        for name, loaded_weight in weights:
            # skip pooling header
            if name.startswith("head."):
                continue

            # post_layernorm is optional
            if (name.startswith("post_layernorm.")
                    and not self.require_post_norm):
                continue

            # omit layers when num_hidden_layers_override is set
            if name.startswith("encoder.layers."):
                layer_idx = int(name.split(".")[2])
                if layer_idx >= layer_count:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or self.use_data_parallel:
                    continue
                name = name.replace(weight_name, param_name)
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
        return loaded_params
