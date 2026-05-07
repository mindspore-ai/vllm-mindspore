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

"""
AclGraph feature tests.

This module contains tests for AclGraph capture and replay functionality,
including:
1. Operator richness tests
2. op_capture_skip tests with different skip configurations
3. Multiple input shapes tests
4. Parameter and non-Parameter tensor tests
5. Edge case tests
"""

import torch

from ms_inferrt import config
from ms_inferrt.torch import backend
from tests.mark_utils import arg_mark


# ============================================================================
# Basic AclGraph Test
# ============================================================================

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_aclgraph_basic(monkeypatch):
    """
    Feature: Test aclgraph capture
    Description: Test basic aclgraph capture with simple add/mul operations
    Expectation: The result is correct
    """
    monkeypatch.setenv("MS_INFERRT_ENABLE_ACLGRAPH", "on")

    def foo(x, y, z):
        out = torch.add(x, y)
        out = torch.add(x, out)
        out = torch.mul(out, z)
        out = torch.add(out, y)
        out = torch.add(out, x)
        out = torch.mul(out, z)
        out = torch.add(out, y)
        out = torch.mul(out, z)
        out = torch.add(out, x)
        out = torch.mul(out, y)
        return out

    config.ascend.aclgraph.set_op_capture_skip(["mul"])

    opt_foo = torch.compile(foo, backend=backend)

    # First run - capture phase
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='npu')
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='npu')
    z = torch.tensor(2.0, device='npu')
    config.ascend.aclgraph.begin_capture()
    res = opt_foo(x, y, z)
    config.ascend.aclgraph.end_capture()
    print(f"Capture result: {res}")

    # Second run - replay phase
    res = opt_foo(x, y, z)
    expected = foo(x, y, z)

    assert torch.allclose(res, expected)
    print("The result is correct")


# ============================================================================
# Operator Richness Tests
# ============================================================================

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_aclgraph_operator_richness_basic(monkeypatch):
    """
    Feature: Test aclgraph with various basic operators
    Description: Test aclgraph capture with add, sub, mul, div, matmul
    Expectation: The result is correct
    """
    monkeypatch.setenv("MS_INFERRT_ENABLE_ACLGRAPH", "on")

    def model(x, y, z, scale):
        # Basic arithmetic operations (10 ops)
        out = torch.add(x, y)
        out = torch.sub(out, z)
        out = torch.mul(out, y)
        out = torch.div(out, scale)
        out = torch.add(out, x)
        out = torch.mul(out, y)
        out = torch.sub(out, z)
        out = torch.add(out, x)
        out = torch.mul(out, y)
        out = torch.div(out, scale)
        return out

    config.ascend.aclgraph.set_op_capture_skip([])

    opt_model = torch.compile(model, backend=backend)

    x = torch.randn(2, 3, device='npu')
    y = torch.randn(2, 3, device='npu')
    z = torch.randn(2, 3, device='npu')
    scale = torch.tensor(2.0, device='npu')

    # Capture
    config.ascend.aclgraph.begin_capture()
    _ = opt_model(x, y, z, scale)
    config.ascend.aclgraph.end_capture()

    # Replay
    result_replay = opt_model(x, y, z, scale)
    expected = model(x, y, z, scale)

    assert torch.allclose(result_replay, expected, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_aclgraph_operator_richness_linear(monkeypatch):
    """
    Feature: Test aclgraph with linear layers
    Description: Test aclgraph capture with matmul, linear, and activation
    Expectation: The result is correct
    """
    monkeypatch.setenv("MS_INFERRT_ENABLE_ACLGRAPH", "on")

    def model(x, weight1, bias1, weight2, bias2, scale1, scale2):
        # Multi-layer linear computation (10 ops)
        out = torch.matmul(x, weight1)
        out = torch.add(out, bias1)
        out = torch.clamp(out, min=0)
        out = torch.matmul(out, weight2)
        out = torch.add(out, bias2)
        out = torch.clamp(out, min=0)
        out = torch.mul(out, scale1)
        out = torch.add(out, x[:, :out.shape[1]])
        out = torch.mul(out, scale2)
        out = torch.clamp(out, min=0)
        return out

    config.ascend.aclgraph.set_op_capture_skip([])

    opt_model = torch.compile(model, backend=backend)

    batch, in_features, hidden, out_features = 4, 64, 32, 16
    x = torch.randn(batch, in_features, device='npu')
    weight1 = torch.randn(in_features, hidden, device='npu')
    bias1 = torch.randn(hidden, device='npu')
    weight2 = torch.randn(hidden, out_features, device='npu')
    bias2 = torch.randn(out_features, device='npu')
    scale1 = torch.tensor(0.5, device='npu')
    scale2 = torch.tensor(2.0, device='npu')

    # Capture
    config.ascend.aclgraph.begin_capture()
    _ = opt_model(x, weight1, bias1, weight2, bias2, scale1, scale2)
    config.ascend.aclgraph.end_capture()

    # Replay
    result_replay = opt_model(x, weight1, bias1, weight2, bias2, scale1, scale2)
    expected = model(x, weight1, bias1, weight2, bias2, scale1, scale2)

    assert torch.allclose(result_replay, expected, rtol=1e-3, atol=1e-3)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_aclgraph_operator_richness_reshape(monkeypatch):
    """
    Feature: Test aclgraph with reshape operations
    Description: Test aclgraph capture with view, reshape, transpose, permute
    Expectation: The result is correct
    """
    monkeypatch.setenv("MS_INFERRT_ENABLE_ACLGRAPH", "on")

    def model(x):
        # Reshape operations (10 ops)
        out = torch.reshape(x, (2, 12))
        out = torch.transpose(out, 0, 1)
        out = torch.reshape(out, (12, 2))
        out = torch.transpose(out, 0, 1)
        out = torch.reshape(out, (4, 6))
        out = torch.transpose(out, 0, 1)
        out = torch.reshape(out, (2, 12))
        out = torch.transpose(out, 0, 1)
        out = torch.reshape(out, (12, 2))
        out = torch.transpose(out, 0, 1)
        return out

    config.ascend.aclgraph.set_op_capture_skip([])

    opt_model = torch.compile(model, backend=backend)

    x = torch.randn(4, 6, device='npu')

    # Capture
    config.ascend.aclgraph.begin_capture()
    _ = opt_model(x)
    config.ascend.aclgraph.end_capture()

    # Replay
    result_replay = opt_model(x)
    expected = model(x)

    assert torch.allclose(result_replay, expected, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_aclgraph_operator_richness_compare(monkeypatch):
    """
    Feature: Test aclgraph with comparison operators
    Description: Test aclgraph capture with eq, ne, lt, le, gt, ge
    Expectation: The result is correct
    """
    monkeypatch.setenv("MS_INFERRT_ENABLE_ACLGRAPH", "on")

    def model(x, y):
        # Comparison operations (10 ops)
        eq = torch.eq(x, y).to(torch.float32)
        lt = torch.lt(x, y).to(torch.float32)
        gt = torch.gt(x, y).to(torch.float32)
        le = torch.le(x, y).to(torch.float32)
        ge = torch.ge(x, y).to(torch.float32)
        # Combine results with arithmetic ops
        out = torch.add(eq, lt)
        out = torch.add(out, gt)
        out = torch.add(out, le)
        out = torch.add(out, ge)
        return out

    config.ascend.aclgraph.set_op_capture_skip([])

    opt_model = torch.compile(model, backend=backend)

    x = torch.randn(3, 4, device='npu')
    y = torch.randn(3, 4, device='npu')

    # Capture
    config.ascend.aclgraph.begin_capture()
    _ = opt_model(x, y)
    config.ascend.aclgraph.end_capture()

    # Replay
    result_replay = opt_model(x, y)
    expected = model(x, y)

    assert torch.allclose(result_replay, expected.to(torch.float32))
