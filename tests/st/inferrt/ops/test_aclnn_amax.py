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
"""Tests for amax via fx_backend."""

import pytest
import torch

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def _make_input(dtype):
    """Create deterministic NPU input for amax tests."""
    if dtype.is_floating_point:
        data = torch.linspace(-6, 6, steps=24, dtype=torch.float32).reshape(2, 3, 4)
        return data.to(dtype).npu()
    data = torch.arange(-12, 12, dtype=dtype).reshape(2, 3, 4)
    return data.npu()


def _make_amax_func(api_name, dim, keepdim):
    """Create an amax frontend variant for torch.compile."""
    if api_name == "aten":
        def aten_amax(x):
            """Call aten.amax.default explicitly."""
            return torch.ops.aten.amax.default(x, dim, keepdim)

        return aten_amax

    if api_name == "torch":
        def torch_amax(x):
            """Call torch.amax."""
            return torch.amax(x, dim=dim, keepdim=keepdim)

        return torch_amax

    if api_name == "torch_no_dim":
        def torch_amax_no_dim(x):
            """Call torch.amax without an explicit dim."""
            return torch.amax(x)

        return torch_amax_no_dim

    if api_name == "method":
        def tensor_amax(x):
            """Call Tensor.amax."""
            return x.amax(dim=dim, keepdim=keepdim)

        return tensor_amax

    if api_name == "method_no_dim":
        def tensor_amax_no_dim(x):
            """Call Tensor.amax without an explicit dim."""
            return x.amax()

        return tensor_amax_no_dim

    raise ValueError(f"Unsupported amax API case: {api_name}")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize(
    "api_name,dtype,dim,keepdim,expect_scalar",
    [
        ("aten", torch.float16, [], False, True),
        ("aten", torch.float32, [], True, False),
        ("torch", torch.bfloat16, [0], False, False),
        ("torch", torch.float16, [-1], True, False),
        ("method", torch.float32, [1, 2], False, False),
        ("method", torch.int32, [1, 2], True, False),
        ("torch_no_dim", torch.float16, None, False, True),
        ("method_no_dim", torch.float32, None, False, True),
    ],
)
def test_amax_fx_backend_frontend_and_arg_variants(
        api_name, dtype, dim, keepdim, expect_scalar):
    """
    Feature: Test amax via fx_backend
    Description: Cover aten, torch, and tensor-method frontend variants
    Expectation: Result matches NPU eager output
    """
    x = _make_input(dtype)
    amax_func = _make_amax_func(api_name, dim, keepdim)

    eager_out = amax_func(x).detach().cpu()
    compiled_func = torch.compile(amax_func, backend=backend, fullgraph=True)
    compiled_out = compiled_func(x).detach().cpu()

    if expect_scalar:
        assert compiled_out.dim() == 0
    AssertRtolEqual(eager_out, compiled_out, prec=1e-3, prec16=2e-3)
