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
"""Tests for log_softmax via fx_backend."""

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.exc import BackendCompilerFailed

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def _make_log_softmax_func(api_name, dim, half_to_float=False, dtype_arg=None):
    """Create a log_softmax frontend variant for torch.compile."""
    if api_name == "aten":
        def aten_log_softmax(x):
            """Call aten._log_softmax.default explicitly."""
            return torch.ops.aten._log_softmax.default(  # pylint: disable=protected-access
                x, dim, half_to_float
            )

        return aten_log_softmax

    if api_name == "torch":
        def torch_log_softmax(x):
            """Call torch.log_softmax."""
            return torch.log_softmax(x, dim=dim, dtype=dtype_arg)

        return torch_log_softmax

    if api_name == "functional":
        def functional_log_softmax(x):
            """Call torch.nn.functional.log_softmax."""
            return F.log_softmax(x, dim=dim, dtype=dtype_arg)

        return functional_log_softmax

    raise ValueError(f"Unsupported log_softmax API case: {api_name}")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize(
    "api_name,dtype,dim,half_to_float,dtype_arg",
    [
        ("aten", torch.float16, -1, False, None),
        ("aten", torch.bfloat16, 1, False, None),
        ("aten", torch.float32, -1, False, None),
        ("torch", torch.float16, 1, False, None),
        ("torch", torch.float16, 0, False, torch.float16),
        ("functional", torch.bfloat16, -1, False, None),
        ("functional", torch.float16, 1, False, torch.float16),
    ],
)
def test_log_softmax_fx_backend_frontend_and_arg_variants(
        api_name, dtype, dim, half_to_float, dtype_arg):
    """
    Feature: Test log_softmax via fx_backend
    Description: Cover aten, torch, and torch.nn.functional frontend variants
    Expectation: Result matches NPU eager output
    """
    torch.manual_seed(0)
    x = torch.randn((3, 4, 5), dtype=dtype).npu()
    log_softmax_func = _make_log_softmax_func(api_name, dim, half_to_float, dtype_arg)

    eager_out = log_softmax_func(x).detach().cpu()
    compiled_func = torch.compile(log_softmax_func, backend=backend, fullgraph=True)
    compiled_out = compiled_func(x).detach().cpu()

    AssertRtolEqual(eager_out, compiled_out, prec=1e-3, prec16=2e-3)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize(
    "api_name,dtype,dim,half_to_float,dtype_arg,err_msg",
    [
        (
            "aten",
            torch.float16,
            0,
            True,
            None,
            "half_to_float=True requires casting the input tensor first",
        ),
        (
            "torch",
            torch.float16,
            0,
            False,
            torch.float32,
            "log_softmax dtype=torch.float32 requires casting the input tensor from torch.float16 first",
        ),
        (
            "functional",
            torch.float16,
            1,
            False,
            torch.float32,
            "log_softmax dtype=torch.float32 requires casting the input tensor from torch.float16 first",
        ),
    ],
)
def test_log_softmax_fx_backend_rejects_dtype_promotion(
        api_name, dtype, dim, half_to_float, dtype_arg, err_msg):
    """
    Feature: Test unsupported log_softmax dtype promotion via fx_backend
    Description: Reject half_to_float and dtype-driven input promotion until cast is lowered
    Expectation: Backend reports unsupported arguments
    """
    x = torch.randn((3, 4, 5), dtype=dtype).npu()
    log_softmax_func = _make_log_softmax_func(api_name, dim, half_to_float, dtype_arg)
    compiled_func = torch.compile(log_softmax_func, backend=backend, fullgraph=True)

    with pytest.raises(BackendCompilerFailed, match=err_msg):
        compiled_func(x)
