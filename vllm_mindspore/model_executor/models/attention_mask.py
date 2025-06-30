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

from mindspore import Tensor, mint
from mindspore import dtype as mstype
from vllm_mindspore.utils import atlas_inference

r"""
PA:ASD-V2.1.5
1.MLA + Q_seqlen =1: no mask.(BF16 mask(0/-10000), FP16 mask(0/-10000)).
2.MLA + Q_seqlen > 1: (MTP/PC/CP), BF16 mask(0/1), FP16 mask (0/-10000)
3.normal + Q_seqlen=1: no mask
4.normal + Q_seqlen > 1: (MTP/PC/CP),BF16 mask(0/-10000), FP16 mask(0/-10000).;

FA:ASD-V2.1.5
1.MLA: not implement;
2.normal: mask BF16(0/1), FP16 mask(0/-10000);
"""


class LowerTriangularMask:
    r"""
    Provide Infer model attention mask.
    Args:
        dtype (ms dtype): The compute type of Infer model.
        max_model_len (int): The max model length of Infer model.
    """

    def __init__(self, dtype, max_model_len):
        self.dtype = dtype
        self.max_model_len = max_model_len

        prefill_mask_coeff = 1.0 if self.dtype == mstype.bfloat16 else -10000.0

        self.prefill_mask = Tensor(np.triu(np.ones(shape=(128, 128), dtype=np.float16), k=1) * \
                                   prefill_mask_coeff, dtype=self.dtype)

        self.decode_mask = Tensor(np.triu(np.ones(shape=(self.max_model_len, self.max_model_len), dtype=np.int8), k=1),
                                  dtype=self.dtype) * -10000

        self.hard_mask = mint.zeros((1, 1), dtype=self.dtype)

    def gen_attention_mask(self, is_prefill, position_ids, query_lens):
        if is_prefill:
            attention_mask = self.prefill_mask
        else:
            if max(query_lens) > 1:
                attention_mask = mint.index_select(self.decode_mask, 0, position_ids)
            else:
                attention_mask = self.hard_mask
        return attention_mask


class MLALowerTriangularMask(LowerTriangularMask):
    r"""
    Provide MLA Infer model attention mask.
    Args:
        dtype (ms dtype): The compute type of Infer model.
        max_model_len (int): The max model length of Infer model.
    """

    def __init__(self, dtype, max_model_len):

        super().__init__(dtype, max_model_len)
        decode_mask_coeff = 1.0 if self.dtype == mstype.bfloat16 else -10000.0
        self.decode_mask = Tensor(np.triu(np.ones(shape=(self.max_model_len, self.max_model_len), dtype=np.int8), k=1),
                                  dtype=self.dtype) * decode_mask_coeff


class PFALowerTriangularMask(LowerTriangularMask):
    r"""
    Provide PFA Infer model attention mask.
    Args:
        dtype (ms dtype): The compute type of Infer model.
        max_model_len (int): The max model length of Infer model.
    """
    def __init__(self, dtype, max_model_len):
        self.dtype = dtype if atlas_inference() else mstype.bool_
        super().__init__(self.dtype, max_model_len)

        self.prefill_mask = Tensor(np.triu(np.ones((2048, 2048), dtype=np.uint8), k=1), dtype=mstype.bool_)

    def gen_attention_mask(self, is_prefill, position_ids, query_lens):
        if is_prefill:
            attention_mask = self.prefill_mask
            if atlas_inference():
                seq_len = (max(query_lens) + 127) // 128 * 128
                lower_triangle_mask = (
                    np.triu(
                        np.ones(shape=(seq_len, seq_len), dtype=np.int8),
                        k=1,
                    )
                )
                prefill_mask = np.expand_dims(lower_triangle_mask, axis=0)
                prefill_mask = np.expand_dims(prefill_mask, axis=1)
                prefill_mask = prefill_mask.astype(np.bool_)
                attention_mask = Tensor(prefill_mask, dtype=mstype.bool_)
        else:
            attention_mask = self.hard_mask
        return attention_mask