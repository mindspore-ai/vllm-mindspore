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
"""Tests for aclnn grouped matmul operation."""
import numpy as np
import torch
import torch_npu

from mrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual

def grouped_matmul_func(x, weight, bias=None, scale=None, offset=None, antiquant_scale=None,
                        antiquant_offset=None, per_token_scale=None, group_list=None,
                        activation_input=None, activation_quant_scale=None, activation_quant_offset=None,
                        split_item=0, group_type=-1, group_list_type=0, act_type=0, tuning_config=None,
                        output_dtype=torch.float16):
    """Execute grouped matmul operation using torch_npu."""
    out = torch_npu.npu_grouped_matmul(
        x, weight, bias=bias, scale=scale, offset=offset,
        antiquant_scale=antiquant_scale, antiquant_offset=antiquant_offset,
        per_token_scale=per_token_scale, group_list=group_list,
        activation_input=activation_input, activation_quant_scale=activation_quant_scale,
        activation_quant_offset=activation_quant_offset, split_item=split_item,
        group_type=group_type, group_list_type=group_list_type, act_type=act_type,
        tuning_config=tuning_config, output_dtype=output_dtype)
    return out


def get_op_func_compiled():
    return torch.compile(grouped_matmul_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0",
           card_mark="onecard", essential_mark="essential")
def test_grouped_matmul_a8w4():
    """
    Feature: Test aclnn grouped matmul with A8W4
    Description: Test aclnn grouped matmul with 1-D inputs
    Expectation: The result is correct
    """

    num_experts = 16
    batch_size = 768
    k_size = 7168
    n_size = 4096
    quant_group_size = 256

    x = torch.randint(-5, 5, (batch_size, k_size), device="npu").to(torch.int8)
    weight = torch.randint(-5, 5, (num_experts, k_size, n_size),
                           dtype=torch.int32, device="npu")
    weight_quant = torch_npu.npu_quantize(
        weight.to(torch.float32), torch.tensor([1.]).npu(), None,
        torch.quint4x2, -1, False)
    bias = torch.zeros((num_experts, n_size), dtype=torch.float32,
                       device="npu").uniform_(-5, 5)
    scale_np = np.random.normal(0, 0.01, (num_experts, 1, n_size)).astype(np.float32)
    per_group_scale = np.ones([num_experts, k_size // quant_group_size, n_size])
    per_group_scale = per_group_scale.astype(np.float32)
    scale_uint32 = (scale_np * per_group_scale).astype(np.float16)
    scale_uint32 = scale_uint32.astype(np.float32)
    scale_uint32.dtype = np.uint32
    scale_uint64 = np.zeros((num_experts, k_size // quant_group_size,
                             n_size * 2), dtype=np.uint32)
    scale_uint64[..., ::2] = scale_uint32
    scale_uint64.dtype = np.int64
    scale = torch.from_numpy(scale_uint64).npu()
    group_list = torch.zeros((num_experts,), dtype=torch.int64, device="npu")
    group_list = group_list.fill_(1)
    per_token_scale = torch.zeros((batch_size, 1), dtype=torch.float32,
                                  device="npu").uniform_()

    compiled_func = get_op_func_compiled()
    output = compiled_func(
        [x.clone()], [weight_quant.clone()], bias=[bias.clone()],
        scale=[scale.clone()], offset=None, antiquant_scale=None,
        antiquant_offset=None, per_token_scale=[per_token_scale.clone()],
        group_list=group_list.clone(), activation_input=None,
        activation_quant_scale=None, activation_quant_offset=None,
        split_item=3, group_type=0, group_list_type=1, act_type=0,
        output_dtype=torch.float16)
    expected = grouped_matmul_func(
        [x.clone()], [weight_quant.clone()], bias=[bias.clone()],
        scale=[scale.clone()], offset=None, antiquant_scale=None,
        antiquant_offset=None, per_token_scale=[per_token_scale.clone()],
        group_list=group_list.clone(), activation_input=None,
        activation_quant_scale=None, activation_quant_offset=None,
        split_item=3, group_type=0, group_list_type=1, act_type=0,
        output_dtype=torch.float16)
    AssertRtolEqual(output[0][:num_experts], expected[0][:num_experts])


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0",
           card_mark="onecard", essential_mark="essential")
def test_grouped_matmul_a8w4_nz():
    """
    Feature: Test aclnn grouped matmul with A8W4
    Description: Test aclnn grouped matmul with 1-D inputs
    Expectation: The result is correct
    """

    num_experts = 8
    batch_size = 768
    k_size = 7168
    n_size = 2048
    quant_group_size = 256

    x = torch.randint(-5, 5, (batch_size, k_size), device="npu").to(torch.int8)
    weight = torch.randint(-5, 5, (num_experts, k_size, n_size),
                           dtype=torch.int32, device="npu")
    weight_nz = torch_npu.npu_format_cast(weight.to(torch.float32), 29)
    weight_quant_nz = torch_npu.npu_quantize(
        weight_nz, torch.tensor([1.]).npu(), None, torch.quint4x2, -1, False)
    bias = torch.zeros((num_experts, n_size), dtype=torch.float32,
                       device="npu").uniform_(-5, 5)
    scale_np = np.random.normal(0, 0.01, (num_experts, 1, n_size)).astype(np.float32)
    per_group_scale = np.ones([num_experts, k_size // quant_group_size, n_size])
    per_group_scale = per_group_scale.astype(np.float32)
    scale_uint32 = (scale_np * per_group_scale).astype(np.float16)
    scale_uint32 = scale_uint32.astype(np.float32)
    scale_uint32.dtype = np.uint32
    scale_uint64 = np.zeros((num_experts, k_size // quant_group_size,
                             n_size * 2), dtype=np.uint32)
    scale_uint64[..., ::2] = scale_uint32
    scale_uint64.dtype = np.int64
    scale = torch.from_numpy(scale_uint64).npu()
    group_list = torch.zeros((num_experts,), dtype=torch.int64, device="npu")
    group_list = group_list.fill_(1)
    per_token_scale = torch.zeros((batch_size, 1), dtype=torch.float32,
                                  device="npu").uniform_()

    compiled_func = get_op_func_compiled()
    output = compiled_func(
        [x.clone()], [weight_quant_nz.clone()], bias=[bias.clone()],
        scale=[scale.clone()], offset=None, antiquant_scale=None,
        antiquant_offset=None, per_token_scale=[per_token_scale.clone()],
        group_list=group_list.clone(), activation_input=None,
        activation_quant_scale=None, activation_quant_offset=None,
        split_item=3, group_type=0, group_list_type=1, act_type=0,
        output_dtype=torch.float16)
    expected = grouped_matmul_func(
        [x.clone()], [weight_quant_nz.clone()], bias=[bias.clone()],
        scale=[scale.clone()], offset=None, antiquant_scale=None,
        antiquant_offset=None, per_token_scale=[per_token_scale.clone()],
        group_list=group_list.clone(), activation_input=None,
        activation_quant_scale=None, activation_quant_offset=None,
        split_item=3, group_type=0, group_list_type=1, act_type=0,
        output_dtype=torch.float16)
    AssertRtolEqual(output[0][:num_experts], expected[0][:num_experts])


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0",
           card_mark="onecard", essential_mark="essential")
def test_grouped_matmul_base():
    """
    Feature: Test aclnn grouped matmul
    Description: Test aclnn grouped matmul with 1-D inputs
    Expectation: The result is correct
    """

    torch.manual_seed(0)
    x = torch.normal(mean=0., std=0.1, size=(1792, 1024),
                     dtype=torch.float16).npu()
    weight = torch.normal(mean=0., std=0.1, size=(2, 1024, 256),
                          dtype=torch.float16).npu()
    group_list = torch.tensor([256, 1792]).npu()
    split_item = 3
    group_type = 0
    compiled_func = get_op_func_compiled()
    output = compiled_func([x.clone()], [weight.clone()], bias=None,
                           group_list=group_list, split_item=split_item,
                           group_type=group_type)
    expected = grouped_matmul_func([x.clone()], [weight.clone()], bias=None,
                                   group_list=group_list, split_item=split_item,
                                   group_type=group_type)
    AssertRtolEqual(output[0], expected[0])


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0",
           card_mark="onecard", essential_mark="essential")
def test_grouped_matmul_multi_dim():
    """
    Feature: Test aclnn grouped matmul
    Description: Test aclnn grouped matmul with 1-D inputs
    Expectation: The result is correct
    """
    torch.manual_seed(0)
    x1 = torch.normal(mean=0., std=0.1, size=(7, 1024, 96),
                      dtype=torch.float16)
    x2 = torch.normal(mean=0., std=0.1, size=(7, 1024, 32),
                      dtype=torch.float16)
    x = [x1, x2]
    weight1 = torch.normal(mean=0., std=0.1, size=(96, 5120),
                           dtype=torch.float16)
    weight2 = torch.normal(mean=0., std=0.1, size=(32, 8192),
                           dtype=torch.float16)
    weight = [weight1, weight2]
    bias1 = torch.normal(mean=0., std=0.1, size=(5120,), dtype=torch.float16)
    bias2 = torch.normal(mean=0., std=0.1, size=(8192,), dtype=torch.float16)
    bias = [bias1, bias2]
    group_list = None
    split_item = 0

    x_clone = []
    weight_clone = []
    bias_clone = []
    for x_i in x:
        x_clone.append(x_i.clone().npu())
    for weight_i in weight:
        weight_clone.append(weight_i.clone().npu())
    for bias_i in bias:
        bias_clone.append(bias_i.clone().npu())

    compiled_func = get_op_func_compiled()
    output = compiled_func(x_clone, weight_clone, bias=bias_clone,
                           group_list=group_list, split_item=split_item,
                           group_type=-1)
    expected = grouped_matmul_func(x_clone, weight_clone, bias=bias_clone,
                                   group_list=group_list, split_item=split_item,
                                   group_type=-1)
    AssertRtolEqual(output[0], expected[0])
    AssertRtolEqual(output[1], expected[1])


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0",
           card_mark="onecard", essential_mark="essential")
def test_grouped_matmul_dynamic():
    """
    Feature: Test aclnn grouped matmul
    Description: Test aclnn grouped matmul with dynamic inputs
    Expectation: The result is correct
    """
    torch.manual_seed(0)
    x1 = torch.randn(size=(1792, 256), dtype=torch.float16)
    weight1 = torch.randn(size=(3, 256, 128), dtype=torch.float16)
    bias1 = torch.randn(size=(3, 128), dtype=torch.float16)
    group_list_npu = torch.tensor([256, 1280, 1792]).npu()
    split_item = 3

    compiled_func = get_op_func_compiled()
    output1 = compiled_func(
        [x1.clone().npu()], [weight1.clone().npu()], bias=[bias1.clone().npu()],
        group_list=group_list_npu, split_item=split_item, group_type=0,
        tuning_config=[1792])
    expected1 = grouped_matmul_func(
        [x1.clone().npu()], [weight1.clone().npu()], bias=[bias1.clone().npu()],
        group_list=group_list_npu, split_item=split_item, group_type=0,
        tuning_config=[1792])
    AssertRtolEqual(output1[0], expected1[0])

    x2 = torch.randn(size=(256, 256), dtype=torch.float16)
    weight2 = torch.randn(size=(3, 256, 256), dtype=torch.float16)
    bias2 = torch.randn(size=(3, 256), dtype=torch.float16)
    group_list_npu = torch.tensor([64, 128, 256]).npu()
    output2 = compiled_func(
        [x2.clone().npu()], [weight2.clone().npu()], bias=[bias2.clone().npu()],
        group_list=group_list_npu, split_item=split_item, group_type=0,
        tuning_config=[256])
    expected2 = grouped_matmul_func(
        [x2.clone().npu()], [weight2.clone().npu()], bias=[bias2.clone().npu()],
        group_list=group_list_npu, split_item=split_item, group_type=0,
        tuning_config=[256])
    AssertRtolEqual(output2[0], expected2[0])
