# Copyright 2024 The vLLM team.
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://wwww.apache.org/licenses/LICENSE-2.0
#
# Unless required by application law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""test case for custom op adv_step_flash"""

import time
import pytest
from vllm_mindspore import npu_ops
import numpy as np
import mindspore as ms
import torch

def benchmark_advance_step_op(sampled_token_ids,
                       input_tokens,
                       input_positions,
                       seq_lens_tensor,
                       num_queries,
                       block_size,
                       block_tables,
                       slot_mapping):
    # update input_tokens
    sampled_token_ids_list = sampled_token_ids[:num_queries].squeeze(-1)
    input_tokens[:num_queries] = sampled_token_ids_list

    # get seq_lens and input_positions
    seq_lens = seq_lens_tensor[:num_queries]
    next_seq_lens = seq_lens + 1
    next_input_pos = next_seq_lens - 1

    # update seq_lens and input_positions
    seq_lens_tensor[:num_queries] = next_seq_lens
    input_positions[:num_queries] = next_input_pos

    block_idx = next_input_pos // block_size
    block_offset = next_input_pos % block_size

    current_block_table = block_tables.gather(1, block_idx.unsqueeze(-1)).squeeze(-1)
    slot_num = current_block_table * block_size + block_offset

    # update slot_mapping
    slot_mapping[:num_queries] = slot_num

def gendata(seed, num_seqs, block_size, block_num, make_tensor):
    """generate inputs"""
    np.random.seed(seed)
    sampled_token_ids = np.random.randint(65536, size=(num_seqs,), dtype=np.int64)
    input_tokens = np.random.randint(100, size=(num_seqs,), dtype=np.int64) # out
    input_positions = np.random.randint(100, size=(num_seqs,), dtype=np.int64) # out
    seq_lens_tensor = np.random.randint(block_size * block_num - 1, size=(num_seqs,), dtype=np.int64) # inplace
    block_tables = np.random.randint(1024, size=(num_seqs, block_num), dtype=np.int64)
    slot_mapping = np.random.randint(100, size=(num_seqs,), dtype=np.int64) # out
    return (make_tensor(sampled_token_ids), \
            make_tensor(input_tokens),      \
            make_tensor(input_positions),   \
            make_tensor(seq_lens_tensor),   \
            make_tensor(block_tables),      \
            make_tensor(slot_mapping))


class TestCustomAdvStepFlash:
    """
    Test Custom op AdvStepFlash.
    """
    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_advstepflash(self):
        """
        test case
        """
        seed = int(time.time() * 1000) % 1000000009
        num_seqs = 256
        block_size = 32
        block_num = 4
        num_queries = num_seqs # no padding
        print("test seed:", seed, flush=True)
        sampled_token_ids1, input_tokens1, input_positions1, seq_lens_tensor1, block_tables1, slot_mapping1 = \
            gendata(seed, num_seqs, block_size, block_num, torch.Tensor)
        benchmark_advance_step_op(sampled_token_ids1,
                                  input_tokens1,
                                  input_positions1,
                                  seq_lens_tensor1,
                                  num_queries,
                                  block_size,
                                  block_tables1,
                                  slot_mapping1)

        sampled_token_ids2, input_tokens2, input_positions2, seq_lens_tensor2, block_tables2, slot_mapping2 = \
            gendata(seed, num_seqs, block_size, block_num, ms.Tensor)
        npu_ops.adv_step_flash(num_seqs=num_seqs,
                            num_queries=num_queries,
                            block_size=block_size,
                            input_tokens=input_tokens2,
                            sampled_token_ids=sampled_token_ids2,
                            input_positions=input_positions2,
                            seq_lens=seq_lens_tensor2,
                            slot_mapping=slot_mapping2,
                            block_tables=block_tables2)

        assert np.allclose(sampled_token_ids1, sampled_token_ids2.asnumpy())
        assert np.allclose(input_tokens1, input_tokens2.asnumpy())
        assert np.allclose(input_positions1, input_positions2.asnumpy())
        assert np.allclose(seq_lens_tensor1, seq_lens_tensor2.asnumpy())
        assert np.allclose(block_tables1, block_tables2.asnumpy())
        assert np.allclose(slot_mapping1, slot_mapping2.asnumpy())
