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
"""Tests for aten.arange via fx_backend."""
import numpy as np
import pytest
import torch

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def get_arange_compiled():
    def custom_op_func(start, end, step):
        return torch.arange(start, end, step, device="npu")
    return torch.compile(custom_op_func, backend=backend)


def get_arange_no_step_compiled():
    def custom_op_func(start, end):
        return torch.arange(start, end, device="npu")
    return torch.compile(custom_op_func, backend=backend)


def get_arange_end_only_compiled():
    def custom_op_func(end):
        return torch.arange(end, device="npu", dtype=torch.int64)
    return torch.compile(custom_op_func, backend=backend)


def get_arange_dtype_compiled(dtype):
    """Compile arange with a fixed dtype."""
    def custom_op_func(start, end, step):
        return torch.arange(start, end, step, device="npu", dtype=dtype)
    return torch.compile(custom_op_func, backend=backend)


def prims_iota_len10_start0_step1_int64(x):
    """Build prims.iota with default start/step and int64 dtype."""
    del x
    return torch.ops.prims.iota.default(
        10,
        start=0,
        step=1,
        dtype=torch.int64,
        device="npu",
        requires_grad=False,
    )


def prims_iota_len10_start5_step1_int64(x):
    """Build prims.iota with a non-zero start and int64 dtype."""
    del x
    return torch.ops.prims.iota.default(
        10,
        start=5,
        step=1,
        dtype=torch.int64,
        device="npu",
        requires_grad=False,
    )


def prims_iota_len10_start2_step3_int64(x):
    """Build prims.iota with step greater than one and int64 dtype."""
    del x
    return torch.ops.prims.iota.default(
        10,
        start=2,
        step=3,
        dtype=torch.int64,
        device="npu",
        requires_grad=False,
    )


def prims_iota_len10_start9_step_neg1_int64(x):
    """Build prims.iota with a negative step and int64 dtype."""
    del x
    return torch.ops.prims.iota.default(
        10,
        start=9,
        step=-1,
        dtype=torch.int64,
        device="npu",
        requires_grad=False,
    )


def prims_iota_len10_start0_step1_int32(x):
    """Build prims.iota with default start/step and int32 dtype."""
    del x
    return torch.ops.prims.iota.default(
        10,
        start=0,
        step=1,
        dtype=torch.int32,
        device="npu",
        requires_grad=False,
    )


def get_prims_iota_reshape_add_compiled():
    """Compile prims.iota followed by reshape and add."""
    def custom_op_func(x):
        iota = torch.ops.prims.iota.default(
            10,
            start=0,
            step=1,
            dtype=torch.int32,
            device="npu",
            requires_grad=False,
        )
        return iota.reshape(2, 5) + x
    return torch.compile(custom_op_func, backend=backend)


def prims_iota_network_lengths(x):
    """Build prims.iota cases observed in network graphs."""
    del x
    device = torch.device("npu", 0)
    buf1324 = torch.ops.prims.iota.default(
        1,
        start=0,
        step=1,
        dtype=torch.int64,
        device=device,
        requires_grad=False,
    )
    buf3713 = torch.ops.prims.iota.default(
        2,
        start=0,
        step=1,
        dtype=torch.int64,
        device=device,
        requires_grad=False,
    )
    buf7140 = torch.ops.prims.iota.default(
        3,
        start=0,
        step=1,
        dtype=torch.int64,
        device=device,
        requires_grad=False,
    )
    return buf1324, buf3713, buf7140


def prims_iota_dynamic_start_step(x, start, step):
    """Build prims.iota with runtime start and step scalar arguments."""
    del x
    return torch.ops.prims.iota.default(
        10,
        start=start,
        step=step,
        dtype=torch.int64,
        device="npu",
        requires_grad=False,
    )


def prims_iota_dynamic_length(x, length):
    """Build prims.iota with runtime length scalar argument."""
    del x
    return torch.ops.prims.iota.default(
        length,
        start=0,
        step=1,
        dtype=torch.int64,
        device="npu",
        requires_grad=False,
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_arange():
    """
    Feature: Test aten.arange via fx_backend
    Description: Basic arange test
    Expectation: Result matches torch reference
    """
    torch.ones(1, device="npu")

    op_func_compiled = get_arange_compiled()
    npu_out = op_func_compiled(0, 10, 1).detach().cpu().numpy()
    cpu_out = np.arange(0, 10, 1, dtype=np.int64)
    AssertRtolEqual(cpu_out, npu_out)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_arange_no_step():
    """
    Feature: Test aten.arange via fx_backend without step
    Description: arange without explicit step (default step=1)
    Expectation: Result matches torch reference
    """
    torch.ones(1, device="npu")

    op_func_compiled = get_arange_no_step_compiled()
    npu_out = op_func_compiled(0, 10).detach().cpu().numpy()
    cpu_out = np.arange(0, 10, dtype=np.int64)
    AssertRtolEqual(cpu_out, npu_out)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_arange_end_only():
    """
    Feature: Test aten.arange via fx_backend with only end argument
    Description: arange(end, device=..., dtype=...) form (single positional arg)
    Expectation: Result matches torch reference
    """
    torch.ones(1, device="npu")

    op_func_compiled = get_arange_end_only_compiled()
    npu_out = op_func_compiled(29).detach().cpu().numpy()
    cpu_out = np.arange(29, dtype=np.int64)
    AssertRtolEqual(cpu_out, npu_out)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.float32, torch.float16])
def test_arange_dtype_variants(dtype):
    """
    Feature: Test aten.arange dtype variants via fx_backend
    Description: arange with integer and floating-point dtypes
    Expectation: Result matches torch reference
    """
    torch.ones(1, device="npu")

    op_func_compiled = get_arange_dtype_compiled(dtype)
    npu_out = op_func_compiled(1, 11, 2).detach().cpu()
    cpu_out = torch.arange(1, 11, 2, dtype=dtype)
    AssertRtolEqual(cpu_out.numpy(), npu_out.numpy())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "op_func,length,start,step,dtype",
    [
        (prims_iota_len10_start0_step1_int64, 10, 0, 1, torch.int64),
        (prims_iota_len10_start5_step1_int64, 10, 5, 1, torch.int64),
        (prims_iota_len10_start2_step3_int64, 10, 2, 3, torch.int64),
        (prims_iota_len10_start9_step_neg1_int64, 10, 9, -1, torch.int64),
        (prims_iota_len10_start0_step1_int32, 10, 0, 1, torch.int32),
    ],
)
def test_prims_iota_default_to_arange(op_func, length, start, step, dtype):
    """
    Feature: Test prims.iota.default via fx_backend
    Description: iota is lowered to Op.iota backed by aclnnArange
    Expectation: Result matches torch reference
    """
    torch.ones(1, device="npu")

    op_func_compiled = torch.compile(op_func, backend=backend)
    x_npu = torch.zeros(length, device="npu", dtype=dtype)
    npu_out = op_func_compiled(x_npu).detach().cpu()
    cpu_out = torch.ops.prims.iota.default(
        length,
        start=start,
        step=step,
        dtype=dtype,
        device="cpu",
        requires_grad=False,
    )
    AssertRtolEqual(cpu_out.numpy(), npu_out.numpy())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_prims_iota_output_consumed_by_reshape_and_add():
    """
    Feature: Test prims.iota.default output consumption via fx_backend
    Description: iota output is consumed by reshape and add
    Expectation: Result matches torch reference
    """
    torch.ones(1, device="npu")

    op_func_compiled = get_prims_iota_reshape_add_compiled()
    x_npu = torch.ones((2, 5), device="npu", dtype=torch.int32)
    npu_out = op_func_compiled(x_npu).detach().cpu()

    cpu_iota = torch.ops.prims.iota.default(
        10,
        start=0,
        step=1,
        dtype=torch.int32,
        device="cpu",
        requires_grad=False,
    )
    cpu_out = cpu_iota.reshape(2, 5) + torch.ones((2, 5), dtype=torch.int32)
    AssertRtolEqual(cpu_out.numpy(), npu_out.numpy())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_prims_iota_network_lengths():
    """
    Feature: Test prims.iota.default network length variants via fx_backend
    Description: iota length 1/2/3 cases observed in network graphs
    Expectation: Result matches torch reference
    """
    torch.ones(1, device="npu")

    op_func_compiled = torch.compile(prims_iota_network_lengths, backend=backend)
    x_npu = torch.empty((), device="npu")
    npu_outs = op_func_compiled(x_npu)
    cpu_outs = [
        torch.ops.prims.iota.default(
            length,
            start=0,
            step=1,
            dtype=torch.int64,
            device="cpu",
            requires_grad=False,
        )
        for length in (1, 2, 3)
    ]

    for cpu_out, npu_out in zip(cpu_outs, npu_outs):
        AssertRtolEqual(cpu_out.numpy(), npu_out.detach().cpu().numpy())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_prims_iota_dynamic_start_step():
    """
    Feature: Test prims.iota.default dynamic scalar start and step
    Description: Reused symbolic arange graph must update runtime start/step scalar values
    Expectation: The result is correct
    """
    torch.ones(1, device="npu")

    op_func_compiled = torch.compile(prims_iota_dynamic_start_step, backend=backend, fullgraph=True)
    x_npu = torch.empty((), device="npu")

    for start, step in ((2, 3), (9, -1), (-4, 2), (100, -3), (0, 1)):
        torch_out = prims_iota_dynamic_start_step(x_npu, start, step)
        inferrt_out = op_func_compiled(x_npu, start, step)
        AssertRtolEqual(torch_out.detach().cpu().numpy(), inferrt_out.detach().cpu().numpy())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_prims_iota_dynamic_length():
    """
    Feature: Test prims.iota.default dynamic scalar length
    Description: Runtime length changes the output tensor shape
    Expectation: The result and output shape are correct
    """
    torch.ones(1, device="npu")

    op_func_compiled = torch.compile(prims_iota_dynamic_length, backend=backend, fullgraph=True)
    x_npu = torch.empty((), device="npu")

    for length in (10, 12, 7):
        torch_out = prims_iota_dynamic_length(x_npu, length)
        inferrt_out = op_func_compiled(x_npu, length)
        assert tuple(inferrt_out.shape) == tuple(torch_out.shape)
        AssertRtolEqual(torch_out.detach().cpu().numpy(), inferrt_out.detach().cpu().numpy())
