# Copyright 2025 Huawei Technologies Co., Ltd
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

"""
infer attention mask.
"""
import numpy as np

import mindspore as ms
from mindspore import Tensor, JitConfig, Model
from mindspore.common import dtype as mstype

class LowerTriangularMask:
    r"""
    Provide Infer model attention mask.
    Args:
        mf_model_config (MF Config): The config of Infer model.

    """

    def __init__(self, mf_model_config):
        compute_dtype = mf_model_config.compute_dtype
        seq_length = mf_model_config.seq_length
        mask_coeff = 1.0 if compute_dtype is mstype.bfloat16 else -10000.0

        self.prefill_mask = Tensor(np.triu(np.ones(shape=(128, 128), dtype=np.float16) * mask_coeff, k=1), dtype=compute_dtype)

        self.decode_mask = Tensor(np.triu(np.ones(shape=(seq_length, seq_length), dtype=np.int8), k=1),
                                  dtype=compute_dtype)

        self.hard_mask = Tensor([0], dtype=compute_dtype).reshape(1, 1)

        self.gather = ms.ops.Gather()

    def gen_attention_mask(self, is_prefill, position_ids, query_lens):
        if is_prefill:
            attention_mask = self.prefill_mask
        else:
            if max(query_lens) > 1:
                attention_mask = self.gather(self.decode_mask, position_ids, 0)
            else:
                attention_mask = self.hard_mask
        return attention_mask
