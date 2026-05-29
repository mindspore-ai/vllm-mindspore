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
"""Tests for ACLNN process-global state lifetime across torch.compile graphs."""

import torch
import torch._dynamo.config as dynamo_config

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark  # pylint: disable=import-error


dynamo_config.cache_size_limit = 64


def _inplace_add_func_a(x, y):
    return x.add_(y)


def _inplace_add_func_b(x, y):
    return x.add_(y)


def _run_compiled_inplace_add(compiled_func, shape):
    x = torch.randn(shape, dtype=torch.float32).npu()
    y = torch.randn(shape, dtype=torch.float32).npu()
    expected = x.clone().add_(y.clone())

    actual = compiled_func(x, y)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_aclnn_runtime_survives_torch_compile_reset():
    """
    Feature: ACLNN runtime lifecycle
    Description: Verify torch.compile graph reset does not finalize ACLNN state required by later compiled graphs
    Expectation: Later compiled execution after reset succeeds and matches eager mode
    """
    try:
        compiled_b = torch.compile(_inplace_add_func_b, backend=backend, fullgraph=True, dynamic=True)
        _run_compiled_inplace_add(compiled_b, (4, 5))

        compiled_a = torch.compile(_inplace_add_func_a, backend=backend, fullgraph=True, dynamic=True)
        _run_compiled_inplace_add(compiled_a, (4, 5))

        del compiled_a
        torch.compiler.reset()

        # Shape change forces a new ACLNN prepare/launch path after reset.
        _run_compiled_inplace_add(compiled_b, (8, 5))
    finally:
        torch.compiler.reset()
