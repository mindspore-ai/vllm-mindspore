# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/lora/ops/torch_ops/lora_ops.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2025 The vLLM team.
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
For punica_npu
"""
from mindspore import mint
from mindspore.ops.auto_generate import grouped_matmul_v4


def einsum_ms(inputs, selected_loras):
    # equal to einsum("bi, boi -> bo", inputs, selected_loras)
    selected_loras = mint.transpose(selected_loras, 1, 2)
    outputs = mint.matmul(inputs.unsqueeze(1), selected_loras).squeeze(1)
    return outputs


def sgmv_expand(inputs,
                lora_b_weights,
                output_tensor,
                b_seq_start_loc,
                seq_len_tensor,
                lora_indices_tensor,
                batches,
                max_seq_length,
                token_nums,
                add_inputs=False):
    group_list = seq_len_tensor
    if len(lora_b_weights.shape) == 4:
        lora_b_weights = lora_b_weights.squeeze(1)
        lora_b_weights = mint.transpose(lora_b_weights, 1, 2)
    weight = lora_b_weights[lora_indices_tensor]
    outputs = grouped_matmul_v4([inputs], [weight],
                                group_list=group_list,
                                split_item=3,
                                group_type=0,
                                group_list_type=1)
    outputs = outputs[0]
    limit = output_tensor.shape[0]
    if outputs.shape[0] == 1 and output_tensor.shape[0] != 1:
        limit = 1
    if add_inputs:
        output_tensor[:, :outputs.shape[1]] += outputs[:limit, :]
    else:
        output_tensor[:, :outputs.shape[1]] = outputs[:limit, :]
    return output_tensor


def bgmv_expand(inputs,
                lora_b_weights,
                output_tensor,
                lora_indices_tensor,
                add_inputs=True):
    selected_loras = lora_b_weights[lora_indices_tensor].astype(
        output_tensor.dtype)
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
    if len(lora_a_weights.shape) == 4:
        lora_a_weights = lora_a_weights.squeeze(1)
        lora_a_weights = mint.transpose(lora_a_weights, 1, 2)
    weight = lora_a_weights[lora_indices_tensor]
    outputs = grouped_matmul_v4([inputs], [weight],
                                group_list=group_list,
                                split_item=3,
                                group_type=0,
                                group_list_type=1)
    outputs = outputs[0]
    output_tensor[:, :outputs.shape[1]] = scaling * outputs[:]
    return output_tensor


def bgmv_shrink(inputs,
                lora_b_weights,
                output_tensor,
                lora_indices_tensor,
                scaling=1.0):
    selected_loras = lora_b_weights[lora_indices_tensor].astype(
        output_tensor.dtype)
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
                      add_inputs=False):
    group_list = seq_len_tensor
    if len(lora_b_weights.shape) == 4:
        lora_b_weights = lora_b_weights.squeeze(1)
        lora_b_weights = mint.transpose(lora_b_weights, 1, 2)
    inputs = inputs.astype(output_tensor.dtype)
    weight = lora_b_weights[lora_indices_tensor]
    outputs = grouped_matmul_v4([inputs], [weight],
                                group_list=group_list,
                                split_item=3,
                                group_type=0,
                                group_list_type=1)
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
                      add_inputs=True):
    selected_loras = lora_b_weights[lora_indices_tensor].astype(
        output_tensor.dtype)
    inputs = inputs.astype(output_tensor.dtype)
    if len(selected_loras.shape) == 4:
        selected_loras = selected_loras.squeeze(1)
    outputs = einsum_ms(inputs, selected_loras)
    if add_inputs:
        output_tensor[:, slice_offset:slice_offset + slice_size] += outputs[:]
    else:
        output_tensor[:, slice_offset:slice_offset + slice_size] = outputs[:]
    return output_tensor
