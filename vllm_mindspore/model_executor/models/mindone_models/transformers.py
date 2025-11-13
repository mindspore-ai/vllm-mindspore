# SPDX-License-Identifier: Apache-2.0
#
# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/model_executor/models/transformers.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2024-2025 The vLLM team.
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
"""Wrapper around `transformers` models"""
from collections.abc import Iterable
from typing import Optional, Union

import mindspore
import numpy as np
import vllm.envs as envs
from mindone.transformers.mindspore_adapter.utils import dtype_to_min
from mindone.transformers.modeling_utils import (ALL_ATTENTION_FUNCTIONS,
                                                 PretrainedConfig,
                                                 PreTrainedModel)
from mindone.transformers.models.auto import AutoModel
from mindspore import mint, nn, ops
from mindspore.common.api import _pynative_executor
from mindspore.ops.auto_generate import PagedAttention, ReshapeAndCache
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from vllm.attention.backends.abstract import AttentionType
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_mindspore.model_executor.models.attention_mask import (
    LowerTriangularMask)
from vllm_mindspore.model_executor.models.mindone_models.base import (
    MindONEModelBase)
from vllm_mindspore.model_executor.models.model_base import AttentionWrapper
from vllm_mindspore.model_executor.utils import (get_model_context,
                                                 set_model_context)
from vllm_mindspore.utils import STR_DTYPE_TO_MS_DTYPE

logger = init_logger(__name__)


class Attention(nn.Cell):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        **ignore_kwargs,
    ) -> None:
        super().__init__()
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Only support DECODER now.")
        if ignore_kwargs:
            print(f"{ignore_kwargs=}")
        if not num_kv_heads:
            num_kv_heads = num_heads
        self.attn_type = attn_type
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.hidden_size_per_partition = num_heads * head_size
        self.kv_hidden_size_per_partition = num_kv_heads * head_size
        input_layout = "BSH"
        self.scale = float(scale)
        pre_tokens = 2147483647
        next_tokens = 2147483647

        self.reshape_and_cache = ReshapeAndCache()
        self.flash_attention = FlashAttentionScore(head_num=num_heads,
                                                   scale_value=self.scale,
                                                   pre_tokens=pre_tokens,
                                                   next_tokens=next_tokens,
                                                   input_layout=input_layout)
        self.paged_attention = PagedAttention(head_num=num_heads,
                                              scale_value=self.scale,
                                              kv_head_num=num_kv_heads)

    # @mindspore.jit
    def construct(
        self,
        query: mindspore.Tensor,
        key: mindspore.Tensor,
        value: mindspore.Tensor,
        key_cache: mindspore.Tensor,
        value_cache: mindspore.Tensor,
        slot_mapping: mindspore.Tensor,
        attn_mask: mindspore.Tensor,
        batch_valid_length: mindspore.Tensor,
        q_seq_lens: mindspore.Tensor,
        block_tables: mindspore.Tensor,
    ) -> mindspore.Tensor:

        # FIXME: pynative accuracy bug,
        # ensure that the input tensors of reshape_and_cache is continuous
        key = key.contiguous()
        value = value.contiguous()

        if get_model_context("is_prefill"):
            bs = batch_valid_length.shape[0]
            _key = ops.cat([
                key[i, -batch_valid_length[i]:, :] for i in range(bs)
            ]).unsqueeze(0)
            _value = ops.cat([
                value[i, -batch_valid_length[i]:, :] for i in range(bs)
            ]).unsqueeze(0)
            # FIXME: pynative accuracy bug,
            # ensure that the input tensors of reshape_and_cache is continuous
            _key = _key.contiguous()
            _value = _value.contiguous()
            cache_out = self.reshape_and_cache(_key, _value, key_cache,
                                               value_cache, slot_mapping)
        else:
            cache_out = self.reshape_and_cache(key, value, key_cache,
                                               value_cache, slot_mapping)

        query = ops.depend(query, cache_out)
        if get_model_context("is_prefill"):
            output = self._run_prefill_forward(query, key, value, attn_mask)
        else:
            output = self._run_decode_forward(query, key_cache, value_cache,
                                              block_tables, batch_valid_length,
                                              attn_mask, q_seq_lens)
        return output

    def _run_prefill_forward(
        self,
        query: mindspore.Tensor,
        key: mindspore.Tensor,
        value: mindspore.Tensor,
        attn_mask: mindspore.Tensor,
    ) -> mindspore.Tensor:
        if attn_mask is not None and attn_mask.dtype == mindspore.bfloat16:
            attn_mask = mint.logical_not(
                attn_mask
            ) if attn_mask.dtype == mindspore.bool_ else attn_mask.bool()
        attn_mask = attn_mask.to(query.dtype)

        _, _, _, output = self.flash_attention(
            query,
            key,
            value,
            None,
            None,
            None,
            attn_mask,
            None,
        )

        return output

    @mindspore.jit(dynamic=1)
    def _run_decode_forward(
        self,
        query: mindspore.Tensor,
        key_cache: mindspore.Tensor,
        value_cache: mindspore.Tensor,
        block_tables: mindspore.Tensor,
        batch_valid_length: mindspore.Tensor,
        attn_mask: mindspore.Tensor,
        q_seq_lens: mindspore.Tensor,
    ) -> mindspore.Tensor:
        """Decode with PagedAttention.

        Args:
            query: Its shape is [batch_size, 1, hidden_size].
            key_cache: Its shape is
                [num_block, block_size, kv_heads_per_partition, head_size].
            value_cache: Its shape is
                [num_block, block_size, kv_heads_per_partition, head_size].
            block_tables: Its shape is [block_size, num_block].
            context_lens: Its shape is [batch_size, ].
        """
        output = self.paged_attention(query, key_cache, value_cache,
                                      block_tables, batch_valid_length, None,
                                      None, attn_mask, q_seq_lens)
        return output


def vllm_flash_attention_forward(
        # Transformers args
        module: nn.Cell,
        query: mindspore.Tensor,
        key: mindspore.Tensor,
        value: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
        # Transformers kwargs
        scaling: Optional[float],
        # vLLM/PA kwargs
        key_cache: tuple[mindspore.Tensor],
        value_cache: tuple[mindspore.Tensor],
        slot_mapping: mindspore.Tensor,
        attn_mask: mindspore.Tensor,
        batch_valid_length: tuple[int],
        q_seq_lens: mindspore.Tensor,
        block_tables: mindspore.Tensor,
        # vLLM kwargs
        attention_instances: dict[int, Attention],
        **kwargs):
    layer_idx = module.layer_idx
    self_attn = attention_instances[layer_idx]
    k_cache = key_cache[layer_idx]
    v_cache = value_cache[layer_idx]

    # BNSD -> BSH
    B, _, S, _ = query.shape
    query = query.swapdims(1, 2).reshape(B, S, -1)
    key = key.swapdims(1, 2).reshape(B, S, -1)
    value = value.swapdims(1, 2).reshape(B, S, -1)

    attn_output = self_attn(query, key, value, k_cache, v_cache, slot_mapping,
                            attn_mask, batch_valid_length, q_seq_lens,
                            block_tables)

    return attn_output, None


ALL_ATTENTION_FUNCTIONS["vllm"] = vllm_flash_attention_forward


def log_replacement(name: str, old_module: nn.Cell, new_module: nn.Cell):
    logger.debug("%s: %s -> %s", name, old_module, new_module)


class TransformersModel(nn.Cell):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        logger.info("Using Transformers backend.")

        config: PretrainedConfig = vllm_config.model_config.hf_config
        cache_config: CacheConfig = vllm_config.cache_config
        model_config: ModelConfig = vllm_config.model_config
        parallel_config: ParallelConfig = vllm_config.parallel_config
        quant_config: QuantizationConfig = vllm_config.quant_config

        self.config = config
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.quant_config = quant_config

        if config.model_type in ["gemma2", "gemma3_text", "cohere2"
                                 ] and not hasattr(config, "sliding_window"):
            config.sliding_window = vllm_config.model_config.hf_text_config.interleaved_sliding_window  #noqa: E501

        # FIXME(Isotr0py): We need to refactor this part in the future to
        # avoid registering an extra model layer, otherwise we will need a
        # weights mapper to rename weights.
        self.model: PreTrainedModel = AutoModel.from_config(
            config,
            attn_implementation="vllm",
            torch_dtype=config.torch_dtype,
            trust_remote_code=model_config.trust_remote_code,
        )

        # Attention layers
        self.attention_instances = self.create_attention_instances()

    def pipeline_parallel(self):
        raise NotImplementedError

    def tensor_parallel(self):
        raise NotImplementedError

    def create_attention_instances(self) -> dict[int, Attention]:
        # init global attention instance
        num_heads = self.config.num_attention_heads
        head_size = getattr(
            self.config, "head_dim",
            self.config.hidden_size // self.config.num_attention_heads)
        num_kv_heads = self.config.num_key_value_heads
        scaling = head_size**-0.5

        # global attention_instances
        return {
            layer_idx:
            Attention(num_heads,
                      head_size,
                      scaling,
                      num_kv_heads=num_kv_heads,
                      prefix=f"model.layers.{layer_idx}.self_attn.attn",
                      attn_type=AttentionType.DECODER)
            for layer_idx in range(self.config.num_hidden_layers)
        }

    def get_input_embeddings(self) -> nn.Cell:
        return self.model.get_input_embeddings()

    def construct(
            self,
            # transformers arguments
            input_ids,
            attention_mask,
            position_ids,
            # PA arguments
            key_cache,
            value_cache,
            slot_mapping,
            attn_mask,
            batch_valid_length,
            q_seq_lens,
            block_tables) -> Union[mindspore.Tensor, IntermediateTensors]:

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            cache_position=None,
            # PA arguments
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            batch_valid_length=batch_valid_length,
            q_seq_lens=q_seq_lens,
            block_tables=block_tables,
            # vLLM arguments
            attention_instances=self.attention_instances,
        )

        hidden_states = model_outputs if isinstance(
            model_outputs, mindspore.Tensor) else model_outputs[0]

        return hidden_states

    def load_weights(
            self, weights: Iterable[tuple[str, mindspore.Tensor]]) -> set[str]:
        raise NotImplementedError


class TransformersForCausalLM(MindONEModelBase):
    embedding_padding_modules = ["lm_head"]
    embedding_modules = ["embed_tokens"
                         ]  # TODO transformers will have a util to get it

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config: PretrainedConfig = vllm_config.model_config.hf_config
        self.config = config

        if self.model_config.dtype in STR_DTYPE_TO_MS_DTYPE:
            self.model_config.dtype = STR_DTYPE_TO_MS_DTYPE[
                self.model_config.dtype]

        self.model = TransformersModel(vllm_config=vllm_config, prefix=prefix)

        self.unpadded_vocab_size = config.vocab_size

        # FIXME: adapt to phi
        has_bias = (config.model_type in ["phi"])
        self.lm_head = nn.Dense(config.hidden_size,
                                config.vocab_size,
                                has_bias=has_bias)

        self.lm_head.weight.set_dtype(self.model_config.dtype)
        if has_bias:
            self.lm_head.bias.set_dtype(self.model_config.dtype)

        self.set_modules({"model": self.model, "lm_head": self.lm_head})

        self.logit_scale = getattr(config, "logit_scale", 1.0)

        self.casual_mask = LowerTriangularMask(
            dtype=self.model_config.dtype,
            max_model_len=self.model_config.max_model_len)
        self.kv_caches = [
            AttentionWrapper() for _ in range(config.num_hidden_layers)
        ]
        compilation_config = vllm_config.compilation_config

        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(config.num_hidden_layers):
            compilation_config.static_forward_context[str(
                i)] = self.kv_caches[i]

    def get_causal_mask(
        self,
        attention_mask: mindspore.Tensor,
        positions: mindspore.Tensor,
        query_lens: mindspore.Tensor,
        is_prefill: bool,
        seq_lens_np: mindspore.Tensor,
    ):
        dtype = self.model_config.dtype

        if is_prefill:
            batch_size, sequence_length = positions.shape
            target_length = sequence_length  # +1
            past_seen_tokens = 0
            cache_position = mint.arange(past_seen_tokens,
                                         past_seen_tokens + sequence_length)

            # generate 4d causal attention mask
            if attention_mask is not None and attention_mask.dim() == 4:
                # In this case we assume that
                # the mask comes already in inverted form
                # and requires no inversion or slicing.
                causal_mask = attention_mask
            else:
                min_dtype = dtype_to_min(dtype)
                causal_mask = mint.ones(
                    (sequence_length, target_length), dtype=dtype) * min_dtype

                diagonal_attend_mask = mint.arange(
                    target_length) > cache_position.reshape(-1, 1)
                causal_mask *= diagonal_attend_mask
                causal_mask = causal_mask[None, None, :, :].broadcast_to(
                    (batch_size, 1, -1, -1))
                if attention_mask is not None:
                    causal_mask = causal_mask.clone(
                    )  # copy to contiguous memory for in-place edit
                    if attention_mask.shape[-1] > target_length:
                        attention_mask = attention_mask[:, :target_length]
                    mask_length = attention_mask.shape[-1]
                    padding_mask = causal_mask[:, :, :, :
                                               mask_length] + attention_mask[
                                                   :,
                                                   None,
                                                   None,
                                                   :,
                                               ]
                    padding_mask = padding_mask == 0
                    causal_mask = causal_mask.to(mindspore.float32)
                    causal_mask[:, :, :, :
                                mask_length] = causal_mask[:, :, :, :
                                                           mask_length].masked_fill(
                                                               padding_mask,
                                                               min_dtype.to(
                                                                   mindspore.
                                                                   float32))
                    causal_mask = causal_mask.to(dtype)
        else:
            causal_mask = self.casual_mask.gen_attention_mask(
                is_prefill, positions.squeeze(-1), query_lens, seq_lens_np)

        return causal_mask

    def forward(self,
                input_ids: mindspore.Tensor,
                positions: mindspore.Tensor,
                intermediate_tensors: IntermediateTensors = None,
                inputs_embeds: mindspore.Tensor = None,
                **kwargs) -> Union[mindspore.Tensor, IntermediateTensors]:
        key_cache, value_cache = self.get_kvcache()

        attn_metadata = get_forward_context().attn_metadata
        # 0.9.1 attn_metadata[layer_name], don't have layer_name here
        # so we just take one by default
        if isinstance(attn_metadata, dict) and '1' in attn_metadata:
            attn_metadata = attn_metadata['1']
        if attn_metadata is None:
            attn_metadata = self._dummy_attention_metadata(
                input_ids, positions)
        seq_lens = attn_metadata.seq_lens
        if isinstance(seq_lens, mindspore.Tensor):
            seq_lens = seq_lens.tolist()

        if not envs.VLLM_USE_V1:
            # V0
            max_query_len = attn_metadata.max_query_len
            # When Mutli-Step is enabled with Chunked-Prefill, prefills and
            # decodes are scheduled together. In the first step, all the
            # prefills turn into decodes and max_query_len will be 1.
            if self.is_multi_step_chunked_prefill and max_query_len == 1:
                query_lens = [1] * len(seq_lens)
            else:
                query_lens = attn_metadata.query_lens

            seq_lens_np = np.array(seq_lens, dtype=np.int32)
            query_lens_np = np.array(query_lens, dtype=np.int32)
            kv_cache_lens = seq_lens_np - query_lens_np
            if attn_metadata.num_decode_tokens == 0 and kv_cache_lens.max(
            ) == 0:
                is_prefill = True
            else:
                is_prefill = False
        else:
            # V1
            is_prefill = attn_metadata.max_context_lens == 0
            query_lens_np = attn_metadata.q_seq_lens_np
            seq_lens_np = attn_metadata.seq_lens_np

        if is_prefill:
            # padding
            bs, cur_max_len = len(seq_lens), max(seq_lens)
            input_ids = ops.stack([
                ops.pad(input_ids[sum(seq_lens[:i]):sum(seq_lens[:i + 1])],
                        (cur_max_len - seq_lens[i], 0)) for i in range(bs)
            ])
            positions = ops.stack([
                ops.pad(positions[sum(seq_lens[:i]):sum(seq_lens[:i + 1])],
                        (cur_max_len - seq_lens[i], 0)) for i in range(bs)
            ])
            attn_mask = ops.stack([
                ops.ones(seq_lens[i]) if cur_max_len == seq_lens[i] else
                ops.cat((ops.zeros(cur_max_len - seq_lens[i]),
                         ops.ones(seq_lens[i]))) for i in range(bs)
            ])
        else:
            bs = input_ids.shape[0]
            input_ids = input_ids.reshape(bs, 1)
            positions = positions.reshape(bs, 1)
            attn_mask = None

        is_prefill = bool(is_prefill)
        q_seq_lens = mindspore.Tensor(query_lens_np, dtype=mindspore.int32)
        position_ids = mindspore.Tensor(positions, dtype=mindspore.int32)
        attn_mask = self.get_causal_mask(attn_mask, positions, query_lens_np,
                                         is_prefill, seq_lens_np)
        input_ids = input_ids.astype(mindspore.int32)
        slot_mapping = attn_metadata.slot_mapping
        batch_valid_length = mindspore.from_numpy(seq_lens_np)
        block_tables = attn_metadata.block_tables

        model_inputs = (input_ids, None, position_ids, key_cache, value_cache,
                        slot_mapping, attn_mask, batch_valid_length,
                        q_seq_lens, block_tables)

        # for dummy_attention_metadata
        if is_prefill and not self.has_prefill_warmup:  #type: ignore
            self.has_prefill_warmup = True

        set_model_context("is_prefill", is_prefill)

        model_output = self.model(*model_inputs)
        model_output = model_output[:, -1, :]

        return model_output

    def compute_logits(
        self,
        hidden_states: mindspore.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[mindspore.Tensor]:
        logits = self.lm_head(hidden_states).float()

        if self.logit_scale != 1.0:
            logits *= self.logit_scale

        _pynative_executor.sync()

        return logits

    def load_weights(self, weights: Iterable[tuple[str, mindspore.Tensor]]):

        # TODO: support parallel weight loading
        model_loaded_weights = {
            n: mindspore.Parameter(mindspore.Tensor(w[:]), requires_grad=False)
            for n, w in weights
        }
        if "lm_head.weight" in model_loaded_weights:
            lm_head_loaded_weights = {
                "weight": model_loaded_weights.pop("lm_head.weight")
            }
        if "lm_head.bias" in model_loaded_weights:
            lm_head_loaded_weights.update(
                {"bias": model_loaded_weights.pop("lm_head.bias")})

        # convert embedding weight if need
        if (isinstance(self.model.model.embed_tokens, nn.Embedding)
                and "model.embed_tokens.weight" in model_loaded_weights):
            model_loaded_weights["model.embed_tokens.embedding_table"] \
                = model_loaded_weights.pop("model.embed_tokens.weight")

        missing_keys, unexpected_keys = mindspore.load_param_into_net(
            self.model, model_loaded_weights)
        if missing_keys or unexpected_keys:
            print(f"model weights loading, \n"
                  f"    missing keys: {missing_keys}, \n"
                  f"    unexpected_keys: {unexpected_keys}")
        else:
            print("load model weights success.")

        if self.config.tie_word_embeddings:
            if isinstance(self.model.model.embed_tokens, nn.Embedding):
                self.lm_head.weight.set_data(
                    self.model.model.embed_tokens.embedding_table.data)
            elif isinstance(self.model.model.embed_tokens, mint.nn.Embedding):
                self.lm_head.weight.set_data(
                    self.model.model.embed_tokens.weight.data)
            else:
                raise ValueError("unexpected embed_tokens class: "
                                 f"{type(self.model.model.embed_tokens)}")
            print("tie lm_head weights success.")
        else:
            m, u = mindspore.load_param_into_net(self.lm_head,
                                                 lm_head_loaded_weights)
            if m or u:
                print("lm_head weights loading, "
                      f"missing keys: {m}, unexpected_keys: {u}")
            else:
                print("load lm_head weights success.")
