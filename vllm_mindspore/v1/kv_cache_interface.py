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

from dataclasses import dataclass

from typing_extensions import Self
from vllm.config import VllmConfig
from vllm.utils import cdiv
from vllm.v1.kv_cache_interface import FullAttentionSpec


@dataclass(frozen=True)
class MLAQuantFullAttentionSpec(FullAttentionSpec):

    fa3_quant: bool = False

    @property
    def type_id(self) -> str:
        return f"mla_quant_full_attention_{self.block_size}" \
               f"_{self.page_size_bytes}"

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes

    @property
    def page_size_bytes(self) -> int:
        """
        The size of a page with `block_size` tokens in bytes.
        fa3_quant_layer k_cache is int8, v_cache is bfloat16.
        no_fa3_quant_layer all k_cache and v_cache are bfloat16.
        """
        coef = 1
        ctkv_nope_dim = 512
        qk_rope_dim = 64
        return coef * self.block_size * self.num_kv_heads * \
                ((ctkv_nope_dim if self.fa3_quant else ctkv_nope_dim * 2) + \
                 qk_rope_dim * 2)

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        To ensure that the kvcache processing logic for fa quantization is
        unified and all goes through UniformTypeKVCacheSpecs, and convenience
        to handle in the _allocate_nz_kv_cache_tensors_fa3 function, false is
        directly assert here.

        The `assert False` is mainly for the scenarios where all layers are
        either faquant or non-faquant. Logically, they can be merged, but for
        the unification of fquant kvcache manage logic, `False` is directly
        returned here.
        """
        assert False  # noqa: B011
