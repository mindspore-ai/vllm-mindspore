# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
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
"""Inference-only MindSpore model compatible with HuggingFace weights."""
from abc import abstractmethod
from typing import Any, Iterable, Optional, Tuple

import mindspore as ms
import numpy as np
import torch
from mindspore import Tensor, mint, mutable


from vllm.logger import init_logger

from vllm_mindspore.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm_mindspore.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm_mindspore.model_executor.models.llama import LlamaForCausalLM
from .tensor_convert import tensor_ms2torch, tensor_torch2ms


type_model_map = {
    "qwen2": Qwen2ForCausalLM,
    "qwen3": Qwen3ForCausalLM,
    "llama": LlamaForCausalLM,
}

logger = init_logger("vllm_mindspore.models")


class LowerTriangularMask:
    r"""
    Provide Infer model attention mask.
    Args:
        dtype (ms dtype): The compute type of Infer model.
        max_model_len (int): The max model length of Infer model.
    """

    def __init__(self, dtype, max_model_len, decode_mask_coeff=-10000.0):
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.cached_mask_len = 8 * 1024
        self.decode_mask_coeff = decode_mask_coeff

        prefill_mask_coeff = 1.0 if self.dtype == ms.bfloat16 else -10000.0
        self.prefill_mask = Tensor(
            np.triu(np.ones(shape=(128, 128), dtype=np.float16), k=1)
            * prefill_mask_coeff,
            dtype=self.dtype,
        )

        self.hard_mask = mint.zeros((1, 1), dtype=dtype)
        self.decode_mask = (
            Tensor(
                np.triu(
                    np.ones(
                        shape=(self.cached_mask_len, self.cached_mask_len),
                        dtype=np.int8,
                    ),
                    k=1,
                ),
                dtype=self.dtype,
            )
            * self.decode_mask_coeff
        )

    def create_mask(self, query_lens_np, seq_lens_np):
        """
        when query_lens_np = [3], seq_lens_np = [6], decode_mask_coeff = 1
        init attention mask
        0 0 0 0 0 0
        0 0 0 0 0 0
        0 0 0 0 0 0
        """
        max_seq_len = seq_lens_np.max().item()
        total_q_len = query_lens_np.sum().item()
        attention_mask = mint.zeros((total_q_len, max_seq_len), dtype=self.dtype)

        req_num = query_lens_np.shape[0]
        current_row = 0
        for i in range(req_num):
            q_len = query_lens_np[i].item()
            current_row += q_len
            # skip row when q_len <= 1, to decrease execute time
            if q_len <= 1:
                continue
            seq_len = seq_lens_np[i].item()
            context_len = seq_len - q_len
            """
            set the right half to 1
            0 0 0 1 1 1
            0 0 0 1 1 1
            0 0 0 1 1 1
            """
            attention_mask[current_row - q_len:current_row,
			               context_len:] = self.decode_mask_coeff
            """
            set the lower triangle of the right half to 0
            0 0 0 0 1 1
            0 0 0 0 0 1
            0 0 0 0 0 0
            """
            right_tensor = attention_mask[current_row - q_len:current_row,
                                          context_len:seq_len]

            # use masked_fill_ to inplace modify attention_mask
            right_tensor.masked_fill_(
                right_tensor.tril() == self.decode_mask_coeff, 0)

        return attention_mask

    def gen_attention_mask(
        self,
        is_prefill: bool,
        position_ids: Tensor,
        query_lens_np: np.ndarray,
        seq_lens_np: np.ndarray,
    ):
        max_query_len = query_lens_np.max()
        max_seq_len = seq_lens_np.max()
        if is_prefill:
            attention_mask = self.prefill_mask
        elif max_query_len > 1:
            if max_seq_len <= self.cached_mask_len:
                attention_mask = mint.index_select(self.decode_mask, 0, position_ids)
            else:
                attention_mask = self.create_mask(query_lens_np, seq_lens_np)
        else:
            attention_mask = self.hard_mask
        return attention_mask


class MindSporeForCausalLM(torch.nn.Module):
    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        ms.set_context(graph_kernel_flags="--disable_pass=gather_pre_rms_norm_fusion")

        model_type = self.config.model_type
        if model_type not in type_model_map:
            raise ValueError(f"Unsupported arch {arch}")
        arch = type_model_map[model_type]
        self.model = arch(config=config, quant_config=quant_config)

        self.casual_mask = LowerTriangularMask(
            self.config.param_dtype, self.config.max_position_embeddings
        )
        self.key_cache = []
        self.value_cache = []

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        self.model.load_weights(weights)

    def get_kvcache(self, forward_batch: ForwardBatch):
        if self.key_cache and self.value_cache:
            return mutable(self.key_cache), mutable(self.value_cache)

        for i in range(self.config.num_hidden_layers):
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(i)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(i)

            self.key_cache.append(tensor_torch2ms(k_cache))
            self.value_cache.append(tensor_torch2ms(v_cache))

        return mutable(self.key_cache), mutable(self.value_cache)

    def prepare_inputs(self, input_ids, positions, forward_batch):
        key_cache, value_cache = self.get_kvcache(forward_batch)

        # Different processing for the mindspore attention operator
        # Without any prefix cache => Use FlashAttentionScore
        # With cache => Use PagedAttention, no matter the query length is 1 or not
        is_prefill = forward_batch.forward_mode.is_extend()
        is_prefill = is_prefill and forward_batch.extend_prefix_lens.sum().item() == 0

        batch_valid_length = forward_batch.seq_lens.cpu().numpy()

        if forward_batch.extend_seq_lens is not None:
            q_seq_lens = forward_batch.extend_seq_lens.cpu().numpy()
        else:
            q_seq_lens = np.ones([forward_batch.batch_size], dtype=np.int32)

        page_size = forward_batch.token_to_kv_pool.page_size
        block_tables = tensor_torch2ms(
            (
                forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : forward_batch.seq_lens.max()
                ][:, ::page_size]
                // page_size
            )
        ).to(ms.int32)

        model_inputs = {}
        model_inputs["input_ids"] = tensor_torch2ms(input_ids).to(ms.int32)
        model_inputs["batch_valid_length"] = ms.Tensor(
            batch_valid_length, dtype=ms.int32
        )
        model_inputs["position_ids"] = tensor_torch2ms(positions)
        model_inputs["q_seq_lens"] = ms.Tensor(q_seq_lens, dtype=ms.int32)
        model_inputs["attention_mask"] = self.casual_mask.gen_attention_mask(
            is_prefill, model_inputs["position_ids"], q_seq_lens, batch_valid_length
        ).contiguous()
        model_inputs["out_cache_loc"] = tensor_torch2ms(forward_batch.out_cache_loc).to(
            ms.int32
        )
        model_inputs["is_prefill"] = is_prefill
        model_inputs["key_cache"] = key_cache
        model_inputs["value_cache"] = value_cache
        model_inputs["block_tables"] = block_tables
        return model_inputs

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tensor:
        # prepare base inputs
        model_inputs = self.prepare_inputs(input_ids, positions, forward_batch)
        # prepare model inputs
        model_inputs = self.model.prepare_inputs(forward_batch, model_inputs)

        logits = self.model(**model_inputs)

        # TODO: npu tensor ms2torch error to be fix, remain issues of torch_npu to get tensor from dlpack
        logits_result = LogitsProcessorOutput(
            next_token_logits=torch.Tensor(logits.asnumpy()).to(input_ids.device)
        )
        return logits_result

