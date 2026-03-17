# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Tests for torch.ops.npu.npu_moe_gating_top_k operation."""
import numpy as np
import pytest
import torch

from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def moe_gating_top_k_numpy(x: torch.Tensor, k: int, bias: torch.Tensor = None, k_group: int = 1,
                           group_count: int = 1, group_select_mode: int = 0, renorm: int = 0,
                           norm_type: int = 0, out_flag: bool = False, routed_scaling_factor: float = 1.0,
                           eps: float = 1e-20) -> tuple:
    """Reference implementation of moe_gating_top_k using numpy."""
    _ = renorm
    dtype = x.dtype
    if dtype != torch.float32:
        x = x.to(dtype=torch.float32)
        if bias is not None:
            bias = bias.to(dtype=torch.float32)

    x_np = x.numpy()
    bias_np = bias.numpy() if bias is not None else None

    if norm_type == 0:
        x_np = np.exp(x_np - np.log(np.sum(np.exp(x_np), axis=-1, keepdims=True)))
    else:
        x_np = 1 / (1 + np.exp(-x_np))

    original_x = x_np
    if bias_np is not None:
        x_np = x_np + bias_np

    if group_count > 1:
        x_np = x_np.reshape(x_np.shape[0], group_count, -1)
        if group_select_mode == 0:
            group_x = np.amax(x_np, axis=-1)
        else:
            group_x = np.partition(x_np, -2, axis=-1)[..., -2:].sum(axis=-1)
        indices = np.argsort(-group_x, axis=-1, kind='stable')[:, :k_group]

        mask = np.ones((x_np.shape[0], group_count), dtype=bool)
        mask[np.arange(x_np.shape[0])[:, None], indices] = False
        x_np = np.where(mask[..., None], float('-inf'), x_np)
        x_np = x_np.reshape(x_np.shape[0], -1)

    indices = np.argsort(-x_np, axis=-1, kind='stable')
    indices = indices[:, :k]
    y = np.take_along_axis(original_x, indices, axis=1)

    if norm_type == 1:
        y = y / (np.sum(y, axis=-1, keepdims=True) + eps)
    y = y * routed_scaling_factor

    if out_flag:
        out = original_x
    else:
        out = None

    y = torch.tensor(y, dtype=dtype)
    return y, indices.astype(np.int32), out


def get_op_func_compiled():
    """Get compiled op function."""
    def custom_op_func(x, k, bias, k_group, group_count, group_select_mode, renorm, norm_type,
                       out_flag, routed_scaling_factor, eps):
        return torch.ops.npu.npu_moe_gating_top_k(x, k, bias=bias, k_group=k_group,
                                                   group_count=group_count,
                                                   group_select_mode=group_select_mode,
                                                   renorm=renorm, norm_type=norm_type,
                                                   out_flag=out_flag,
                                                   routed_scaling_factor=routed_scaling_factor,
                                                   eps=eps)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("n,k", [(8, 6), (128, 16), (1002, 1)])
@pytest.mark.parametrize("group_select_mode", [0, 1])
@pytest.mark.parametrize("norm_type", [0, 1])
@pytest.mark.parametrize("routed_scaling_factor", [0.5, 1.5])
def test_moe_gating_top_k(dtype, n, k, group_select_mode, norm_type, routed_scaling_factor):
    """
    Feature: Test aclnn moe_gating_top_k
    Description: Test aclnn moe_gating_top_k with various combinations of dtype, batch size, k,
                 group_select_mode, norm_type, and routed_scaling_factor
    Expectation: The result is correct
    """
    x = np.random.uniform(-2, 2, (n, 256)).astype(np.float32)
    bias = np.random.uniform(-2, 2, (256,)).astype(np.float32)

    x_tensor = torch.tensor(x, dtype=dtype)
    bias_tensor = torch.tensor(bias, dtype=dtype)

    k_group = 4
    group_count = 8
    renorm = 0
    out_flag = False
    eps = 1e-20

    y_golden, expert_idx_golden, _ = moe_gating_top_k_numpy(
        x_tensor, k, bias=bias_tensor, k_group=k_group, group_count=group_count,
        group_select_mode=group_select_mode, renorm=renorm, norm_type=norm_type,
        out_flag=out_flag, routed_scaling_factor=routed_scaling_factor, eps=eps)

    x_npu = x_tensor.npu()
    bias_npu = bias_tensor.npu()

    op_func_compiled = get_op_func_compiled()
    y, expert_idx, _ = op_func_compiled(x_npu, k, bias_npu, k_group, group_count,
                                         group_select_mode, renorm, norm_type,
                                         out_flag, routed_scaling_factor, eps)

    AssertRtolEqual(y.detach().cpu(), y_golden)
    AssertRtolEqual(expert_idx.detach().cpu(), torch.from_numpy(expert_idx_golden))
