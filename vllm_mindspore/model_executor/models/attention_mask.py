# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologies Co., Ltd.
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
"""
infer attention mask.
"""
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import mint

# yapf conflicts with isort
# yapf: disable  # noqa: ERA001

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

# yapf: enable  # noqa: ERA001


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

        prefill_mask_coeff = 1.0 if self.dtype == mstype.bfloat16 else -10000.0
        self.prefill_mask = Tensor(
            np.triu(np.ones(shape=(128, 128), dtype=np.float16), k=1) *
            prefill_mask_coeff,
            dtype=self.dtype)

        self.hard_mask = mint.zeros((1, 1), dtype=dtype)
        self.decode_mask = Tensor(np.triu(np.ones(
            shape=(self.cached_mask_len, self.cached_mask_len), dtype=np.int8),
                                          k=1),
                                  dtype=self.dtype) * self.decode_mask_coeff

    def create_mask(self, query_lens_np, seq_lens_np):
        '''
        when query_lens_np = [3], seq_lens_np = [6], decode_mask_coeff = 1
        init attention mask
        0 0 0 0 0 0
        0 0 0 0 0 0
        0 0 0 0 0 0
        '''
        max_seq_len = seq_lens_np.max().item()
        total_q_len = query_lens_np.sum().item()
        attention_mask = mint.zeros((total_q_len, max_seq_len),
                                    dtype=self.dtype)

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
            '''
            set the right half to 1
            0 0 0 1 1 1
            0 0 0 1 1 1
            0 0 0 1 1 1
            '''
            attention_mask[current_row - q_len:current_row,
                           context_len:] = self.decode_mask_coeff
            '''
            set the lower triangle of the right half to 0
            0 0 0 0 1 1
            0 0 0 0 0 1
            0 0 0 0 0 0
            '''
            right_tensor = attention_mask[current_row - q_len:current_row,
                                          context_len:seq_len]
            # use masked_fill_ to inplace modify attention_mask
            right_tensor.masked_fill_(
                right_tensor.tril() == self.decode_mask_coeff, 0)

        return attention_mask

    def gen_attention_mask(self, is_prefill: bool, position_ids: Tensor,
                           query_lens_np: np.ndarray, seq_lens_np: np.ndarray):
        max_query_len = query_lens_np.max()
        max_seq_len = seq_lens_np.max()
        if is_prefill:
            attention_mask = self.prefill_mask
        elif max_query_len > 1:
            if max_seq_len <= self.cached_mask_len:
                attention_mask = mint.index_select(self.decode_mask, 0,
                                                   position_ids)
            else:
                attention_mask = self.create_mask(query_lens_np, seq_lens_np)
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
        decode_mask_coeff = 1.0 if dtype == mstype.bfloat16 else -10000.0
        super().__init__(dtype, max_model_len, decode_mask_coeff)


class MultiModalLowerTriangularMask(LowerTriangularMask):
    r"""
    Provide multi modal Infer model attention mask.
    Args:
        dtype (ms dtype): The compute type of Infer model.
        max_model_len (int): The max model length of Infer model.
    """

    def gen_attention_mask(self, is_prefill, position_ids, query_lens_np,
                           seq_lens_np):
        max_query_len = query_lens_np.max()
        max_seq_len = seq_lens_np.max()
        if is_prefill:
            attention_mask = self.prefill_mask
        elif max_query_len > 1:
            if max_seq_len <= self.cached_mask_len:
                mm_position_ids_list = []
                for i in range(len(seq_lens_np)):
                    mm_position_ids_list.append(
                        np.arange(seq_lens_np[i] - query_lens_np[i],
                                  seq_lens_np[i]))
                mm_position_ids = np.concatenate(mm_position_ids_list)
                mm_position_ids = Tensor(mm_position_ids,
                                         dtype=position_ids.dtype)
                attention_mask = mint.index_select(self.decode_mask, 0,
                                                   mm_position_ids)
            else:
                attention_mask = self.create_mask(query_lens_np, seq_lens_np)
        else:
            attention_mask = self.hard_mask
        return attention_mask
