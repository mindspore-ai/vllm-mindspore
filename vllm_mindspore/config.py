#!/usr/bin/env python3
# encoding: utf-8
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

from vllm.logger import init_logger
from vllm_mindspore.utils import is_mindformers_model_backend, is_use_mla

logger = init_logger(__name__)

def get_head_size(self) -> int:
    if hasattr(self.hf_text_config, "model_type") and (
        self.hf_text_config.model_type in ("deepseek_v2", "deepseek_v3")
    ):

        if is_mindformers_model_backend():
            qk_rope_head_dim = getattr(self.hf_text_config, "qk_rope_head_dim", 0)
            return self.hf_text_config.kv_lora_rank + qk_rope_head_dim

        # FlashAttention supports only head_size 32, 64, 128, 256,
        # we need to pad head_size 192 to 256
        return 256

    if self.is_attention_free:
        return 0

    if hasattr(self.hf_text_config, "head_dim"):
        return self.hf_text_config.head_dim
    # FIXME(woosuk): This may not be true for all models.
    return self.hf_text_config.hidden_size // self.hf_text_config.num_attention_heads


def _verify_quantization(self) -> None:
    # Donnot verify now.
    return


def get_num_kv_heads(self, parallel_config: "ParallelConfig") -> int:
    """Returns the number of KV heads per Device."""

    if is_use_mla(self):
        return 1

    total_num_kv_heads = self.get_total_num_kv_heads()
    return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)


def _verify_args(self) -> None:
    if (self.max_num_batched_tokens < self.max_model_len
            and not self.chunked_prefill_enabled):
        logger.warning(
            f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
            f"smaller than max_model_len ({self.max_model_len}). "
            "This effectively limits the maximum sequence length to "
            "max_num_batched_tokens and makes vLLM reject longer "
            "sequences. Please increase max_num_batched_tokens or "
            "decrease max_model_len.")

    if self.max_num_batched_tokens < self.max_num_seqs:
        raise ValueError(
            f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
            "be greater than or equal to max_num_seqs "
            f"({self.max_num_seqs}).")

    if self.num_lookahead_slots < 0:
        raise ValueError(
            "num_lookahead_slots "
            f"({self.num_lookahead_slots}) must be greater than or "
            "equal to 0.")

    if self.num_scheduler_steps < 1:
        raise ValueError(
            "num_scheduler_steps "
            f"({self.num_scheduler_steps}) must be greater than or "
            "equal to 1.")

def _verify_cache_args(self) -> None:
    if self.gpu_memory_utilization > 1.0:
        raise ValueError(
            "GPU memory utilization must be less than 1.0. Got "
            f"{self.gpu_memory_utilization}.")

