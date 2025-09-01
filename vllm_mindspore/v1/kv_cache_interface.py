# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/v1/kv_cache_interface.py
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

from vllm.config import VllmConfig
from vllm.utils import cdiv
from vllm.v1.kv_cache_interface import AttentionSpec


class MLAQuantFullAttentionSpec(AttentionSpec):

    @property
    def type_id(self) -> str:
        return f"mla_quant_full_attention_{self.block_size}" \
               f"_{self.page_size_bytes}"

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes

    @property
    def page_size_bytes(self) -> int:
        # For MLA we only store a single latent vector
        coef = 1 if self.use_mla else 2
        assert self.head_size == 576
        ctkv_nope_dim = 512
        qk_rope_dim = 64
        return coef * self.block_size * self.num_kv_heads * (ctkv_nope_dim +
                                                             qk_rope_dim * 2)
