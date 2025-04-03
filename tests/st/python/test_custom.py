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

import mindspore as ms
from vllm_mindspore import npu_ops
import numpy as np
from mindspore import Tensor

# TODO refactor this case to run on ci
def testcase():
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
    in_block_tables = Tensor(np.load("data/block_tables.npy"))
    in_input_positions = Tensor(np.load("data/input_positions.npy"))
    in_input_tokens = Tensor(np.load("data/input_tokens.npy"))
    in_sampled_token_ids = Tensor(np.load("data/sampled_token_ids.npy"))
    in_seq_lens_tensor = Tensor(np.load("data/seq_lens_tensor.npy"))
    in_slot_mapping = Tensor(np.load("data/slot_mapping.npy"))
    num_seqs = 256
    num_queries = 256
    block_size = 32
    npu_ops.adv_step_flash(num_seqs=num_seqs,
                           num_queries=num_queries,
                           block_size=block_size,
                           input_tokens=in_input_tokens,
                           sampled_token_ids=in_sampled_token_ids,
                           input_positions=in_input_positions,
                           seq_lens=in_seq_lens_tensor,
                           slot_mapping=in_slot_mapping,
                           block_tables=in_block_tables)

    out_block_tables = np.load("data/o_block_tables.npy").astype(np.int32)
    out_input_positions = np.load("data/o_input_positions.npy").astype(np.int32)
    out_input_tokens = np.load("data/o_input_tokens.npy").astype(np.int32)
    out_sampled_token_ids = np.load("data/o_sampled_token_ids.npy").astype(np.int32)
    out_seq_lens_tensor = np.load("data/o_seq_lens_tensor.npy").astype(np.int32)
    out_slot_mapping = np.load("data/o_slot_mapping.npy").astype(np.int32)
    assert np.allclose(in_block_tables, out_block_tables)
    assert np.allclose(in_input_positions, out_input_positions)
    assert np.allclose(in_input_tokens, out_input_tokens)
    assert np.allclose(in_sampled_token_ids, out_sampled_token_ids)
    assert np.allclose(in_seq_lens_tensor, out_seq_lens_tensor)
    assert np.allclose(in_slot_mapping, out_slot_mapping)
    print("passed.")

if __name__ == "__main__":
    testcase()
