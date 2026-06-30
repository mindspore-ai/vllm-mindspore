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

"""Tests for aclnn inplace_copy operation (tensor.copy_)."""
import numpy as np
import pytest
import torch

from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def inplace_copy_op(dst, src):
    """op function for inplace copy: dst.copy_(src)"""
    dst.copy_(src)
    return dst


def inplace_copy_op_dynamic(dst, src):
    """op function for inplace copy with dynamic shape"""
    b = dst.size(0)
    dst[:b].copy_(src[:b])
    return dst


def inplace_copy_non_contiguous_op(dst, src):
    """op function for inplace copy on non-contiguous tensor"""
    dst_permuted = dst.permute(1, 0)
    src_permuted = src.permute(1, 0)
    dst_permuted.copy_(src_permuted)
    return dst_permuted


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8, 16), (64, 128), (256, 512)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_inplace_copy_tensor_tensor(shape, dtype):
    """
    Feature: Test aclnn inplace_copy
    Description: Test inplace_copy with fp32/fp16 inputs and various shapes
    Expectation: The result is correct and dst is modified in-place
    """
    compiled_op = torch.compile(inplace_copy_op, backend=backend)
    prec = 0.001 if dtype == np.float16 else 0.0001

    cpu_dst = np.random.uniform(-1, 1, shape).astype(dtype)
    cpu_src = np.random.uniform(0.5, 2, shape).astype(dtype)

    npu_dst = torch.from_numpy(cpu_dst).npu()
    npu_src = torch.from_numpy(cpu_src).npu()

    # CPU reference: clone to preserve original for comparison
    cpu_dst_ref = torch.from_numpy(cpu_dst.copy())
    cpu_src_ref = torch.from_numpy(cpu_src.copy())
    cpu_output = inplace_copy_op(cpu_dst_ref, cpu_src_ref)

    npu_output = compiled_op(npu_dst, npu_src)

    AssertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy(), prec)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(4, 8), (16, 32), (2, 4, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_inplace_copy_different_shapes(dtype, shape):
    """
    Feature: Test aclnn inplace_copy with different shapes and dtypes
    Description: Test inplace_copy with 2D and 3D shapes across fp32/fp16/bf16
    Expectation: The result is correct
    """
    compiled_op = torch.compile(inplace_copy_op, backend=backend)
    prec = 0.001 if dtype in (torch.float16, torch.bfloat16) else 0.0001

    dst_npu = torch.randn(shape, dtype=dtype).npu()
    src_npu = torch.randn(shape, dtype=dtype).npu()

    # Eager reference
    dst_ref = dst_npu.clone()
    src_ref = src_npu.clone()
    expected = inplace_copy_op(dst_ref, src_ref)

    npu_output = compiled_op(dst_npu, src_npu)
    AssertRtolEqual(expected.detach().cpu().numpy(), npu_output.detach().cpu().numpy(), prec)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [(8,), (4, 8), (16, 32), (64, 128), (2, 4, 8)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_inplace_copy_dynamic(shape, dtype):
    """
    Feature: Test aclnn inplace_copy with dynamic shape
    Description: Test inplace_copy with dynamic slicing pattern
    Expectation: The result is correct
    """
    compiled_op = torch.compile(inplace_copy_op_dynamic, backend=backend)
    prec = 0.001 if dtype == np.float16 else 0.0001

    cpu_dst = np.random.uniform(-1, 1, shape).astype(dtype)
    cpu_src = np.random.uniform(0.5, 2, shape).astype(dtype)

    npu_dst = torch.from_numpy(cpu_dst).npu()
    npu_src = torch.from_numpy(cpu_src).npu()

    cpu_dst_ref = torch.from_numpy(cpu_dst.copy())
    cpu_src_ref = torch.from_numpy(cpu_src.copy())
    cpu_output = inplace_copy_op_dynamic(cpu_dst_ref, cpu_src_ref).detach().numpy()

    npu_output = compiled_op(npu_dst, npu_src).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output, prec)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_inplace_copy_non_contiguous(dtype):
    """
    Feature: Test aclnn inplace_copy on non-contiguous tensor
    Description: Test inplace_copy where dst is a permuted (non-contiguous) view
    Expectation: The result is correct
    """
    shape = (4, 8)
    compiled_op = torch.compile(inplace_copy_non_contiguous_op, backend=backend)
    prec = 0.001 if dtype in (torch.float16, torch.bfloat16) else 0.0001

    dst_npu = torch.randn(shape, dtype=dtype).npu()
    src_npu = torch.randn(shape, dtype=dtype).npu()

    # Eager reference
    dst_ref = dst_npu.clone()
    src_ref = src_npu.clone()
    expected = inplace_copy_non_contiguous_op(dst_ref, src_ref)

    npu_output = compiled_op(dst_npu, src_npu)
    AssertRtolEqual(expected.detach().cpu().numpy(), npu_output.detach().cpu().numpy(), prec)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 8), (2, 4, 8), (1, 8, 16, 32)])
def test_inplace_copy_edge_shapes(dtype, shape):
    """
    Feature: Test aclnn inplace_copy with edge-case shapes
    Description: Test inplace_copy with shapes containing dim=1 and higher dimensions
    Expectation: The result is correct
    """
    compiled_op = torch.compile(inplace_copy_op, backend=backend)
    prec = 0.001 if dtype in (torch.float16, torch.bfloat16) else 0.0001

    dst_npu = torch.randn(shape, dtype=dtype).npu()
    src_npu = torch.randn(shape, dtype=dtype).npu()

    dst_ref = dst_npu.clone()
    src_ref = src_npu.clone()
    expected = inplace_copy_op(dst_ref, src_ref)

    npu_output = compiled_op(dst_npu, src_npu)
    AssertRtolEqual(expected.detach().cpu().numpy(), npu_output.detach().cpu().numpy(), prec)
