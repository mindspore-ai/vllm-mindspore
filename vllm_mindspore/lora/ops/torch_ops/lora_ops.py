#!/usr/bin/env python3
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
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
# ============================================================================

"""
For punica_npu
"""
from mindspore import mint
from mindspore.ops.auto_generate import grouped_matmul_v4

def einsum_ms(inputs, selected_loras):
    # mint.einsum("bi, boi -> bo", inputs, selected_loras)
    selected_loras = mint.transpose(selected_loras, 1, 2)
    outputs = mint.matmul(inputs.unsqueeze(1), selected_loras).squeeze(1)
    return outputs

def sort_lora_by_token_count(lora_indices_tensor, seq_len_tensor):
    unique_ids = mint.unique(lora_indices_tensor)
    token_sums = []
    for uid in unique_ids:
        mask = (lora_indices_tensor == uid)
        total_tokens = mint.sum(seq_len_tensor[mask])
        token_sums.append(total_tokens)
    token_sums_tensor = mint.stack(token_sums)
    sorted_counts, sort_indices = mint.sort(token_sums_tensor, descending=True)
    sorted_ids = unique_ids[sort_indices]
    return sorted_ids, sorted_counts

def sgmv_expand(inputs,
                lora_b_weights,
                output_tensor,
                b_seq_start_loc,
                seq_len_tensor,
                lora_indices_tensor,
                batches,
                max_seq_length,
                token_nums,
                add_inputs = False):
    exploded_indices = mint.repeat_interleave(lora_indices_tensor,
                                              seq_len_tensor)

    return bgmv_expand(inputs, lora_b_weights, output_tensor, exploded_indices,
                add_inputs)


def bgmv_expand(inputs,
                lora_b_weights,
                output_tensor,
                lora_indices_tensor,
                add_inputs = True):
    selected_loras = lora_b_weights[lora_indices_tensor].astype(output_tensor.dtype)
    inputs = inputs.astype(output_tensor.dtype)
    if len(selected_loras.shape) == 4:
        selected_loras = selected_loras.squeeze(1)
    outputs = einsum_ms(inputs, selected_loras)
    limit = output_tensor.shape[0]
    if outputs.shape[0] == 1 and output_tensor.shape[0] != 1:
        limit = 1
    if add_inputs:
        output_tensor[:, :outputs.shape[1]] += outputs[:limit, :]
    else:
        output_tensor[:, :outputs.shape[1]] = outputs[:limit, :]
    return output_tensor


def sgmv_shrink(
    inputs,
    lora_a_weights,
    output_tensor,
    b_seq_start_loc,
    seq_len_tensor,
    lora_indices_tensor,
    batches,
    max_seq_length,
    token_nums,
    scaling,
):  
    group_list = seq_len_tensor
    if (lora_indices_tensor.unique().shape[0] != lora_indices_tensor.shape[0]):
        sorted_ids, sorted_counts = sort_lora_by_token_count(lora_indices_tensor, seq_len_tensor)
        group_list = sorted_counts
    if lora_a_weights.shape[0] != group_list.shape[0]:
        new_tensor = mint.zeros(lora_a_weights.shape[0], dtype=group_list.dtype)
        new_tensor[:group_list.size(0)] = group_list
        group_list = new_tensor
    if len(lora_a_weights.shape) == 4:
        lora_a_weights = lora_a_weights.squeeze(1)
        lora_a_weights = mint.transpose(lora_a_weights, 1, 2)
    outputs = grouped_matmul_v4([inputs], [lora_a_weights], group_list=group_list, split_item=3, group_type=0, group_list_type=1)
    outputs = outputs[0]
    output_tensor[:, :outputs.shape[1]] = scaling * outputs[:]
    return output_tensor


def bgmv_shrink(inputs,
                lora_b_weights,
                output_tensor,
                lora_indices_tensor,
                scaling = 1.0):
    selected_loras = lora_b_weights[lora_indices_tensor].astype(output_tensor.dtype)
    inputs = inputs.astype(output_tensor.dtype)
    if len(selected_loras.shape) == 4:
        selected_loras = selected_loras.squeeze(1)
    outputs = einsum_ms(inputs, selected_loras)
    output_tensor[:, :outputs.shape[1]] = scaling * outputs[:]
    return output_tensor


def sgmv_expand_slice(inputs,
                      lora_b_weights,
                      output_tensor,
                      b_seq_start_loc,
                      seq_len_tensor,
                      lora_indices_tensor,
                      batches,
                      max_seq_length,
                      token_nums,
                      slice_offset,
                      slice_size,
                      add_inputs = False):
    group_list = seq_len_tensor
    if (lora_indices_tensor.unique().shape[0] != lora_indices_tensor.shape[0]):
        sorted_ids, sorted_counts = sort_lora_by_token_count(lora_indices_tensor, seq_len_tensor)
        group_list = sorted_counts
    if lora_b_weights.shape[0] != group_list.shape[0]:
        new_tensor = mint.zeros(lora_b_weights.shape[0], dtype=group_list.dtype)
        new_tensor[:group_list.size(0)] = group_list
        group_list = new_tensor
    if len(lora_b_weights.shape) == 4:
        lora_b_weights = lora_b_weights.squeeze(1)
        lora_b_weights = mint.transpose(lora_b_weights, 1, 2)
    inputs = inputs.astype(output_tensor.dtype)
    outputs = grouped_matmul_v4([inputs], [lora_b_weights], group_list=group_list, split_item=3, group_type=0, group_list_type=1)
    outputs = outputs[0]
    if add_inputs:
        output_tensor[:, slice_offset:slice_offset + slice_size] += outputs[:]
    else:
        output_tensor[:, slice_offset:slice_offset + slice_size] = outputs[:]
    return output_tensor


def bgmv_expand_slice(inputs,
                      lora_b_weights,
                      output_tensor,
                      lora_indices_tensor,
                      slice_offset,
                      slice_size,
                      add_inputs = True):
    selected_loras = lora_b_weights[lora_indices_tensor].astype(output_tensor.dtype)
    inputs = inputs.astype(output_tensor.dtype)
    if len(selected_loras.shape) == 4:
        selected_loras = selected_loras.squeeze(1)
    outputs = einsum_ms(inputs, selected_loras)
    if add_inputs:
        output_tensor[:, slice_offset:slice_offset + slice_size] += outputs[:]
    else:
        output_tensor[:, slice_offset:slice_offset + slice_size] = outputs[:]
    return output_tensor