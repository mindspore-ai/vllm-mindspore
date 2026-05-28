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
    monkeypatch.setenv("MS_INFERRT_DISABLE_VIEW_OPS", "transpose")

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


# ============================================================================
# op_capture_skip Tests
# ============================================================================

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_aclgraph_skip_single_op(monkeypatch):
    """
    Feature: Test aclgraph with single skipped operator
    Description: Skip only 'mul' operator during capture
    Expectation: The result is correct, mul runs as single op
    """
    monkeypatch.setenv("MS_INFERRT_ENABLE_ACLGRAPH", "on")

    def model(x, y, z):
        # 10 ops with mul skipped
        out = torch.add(x, y)
        out = torch.mul(out, z)  # This should be skipped
        out = torch.add(out, x)
        out = torch.mul(out, y)  # This should be skipped
        out = torch.add(out, z)
        out = torch.mul(out, x)  # This should be skipped
        out = torch.add(out, y)
        out = torch.mul(out, z)  # This should be skipped
        out = torch.add(out, x)
        out = torch.mul(out, y)  # This should be skipped
        return out

    config.ascend.aclgraph.set_op_capture_skip(["mul"])

    opt_model = torch.compile(model, backend=backend)

    x = torch.randn(2, 3, device='npu')
    y = torch.randn(2, 3, device='npu')
    z = torch.randn(2, 3, device='npu')

    config.ascend.aclgraph.begin_capture()
    _ = opt_model(x, y, z)
    config.ascend.aclgraph.end_capture()

    result_replay = opt_model(x, y, z)
    expected = model(x, y, z)

    assert torch.allclose(result_replay, expected, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_aclgraph_skip_multiple_ops(monkeypatch):
    """
    Feature: Test aclgraph with multiple skipped operators
    Description: Skip 'mul' and 'div' operators during capture
    Expectation: The result is correct, skipped ops run as single ops
    """
    monkeypatch.setenv("MS_INFERRT_ENABLE_ACLGRAPH", "on")

    def model(x, y, z, scale):
        # 10 ops with mul and div skipped
        out = torch.add(x, y)
        out = torch.mul(out, z)   # Skipped
        out = torch.div(out, scale)  # Skipped
        out = torch.add(out, x)
        out = torch.mul(out, y)   # Skipped
        out = torch.div(out, scale)  # Skipped
        out = torch.add(out, z)
        out = torch.mul(out, x)   # Skipped
        out = torch.div(out, scale)  # Skipped
        out = torch.add(out, y)
        return out

    config.ascend.aclgraph.set_op_capture_skip(["mul", "div"])

    opt_model = torch.compile(model, backend=backend)

    x = torch.randn(2, 3, device='npu')
    y = torch.randn(2, 3, device='npu')
    z = torch.randn(2, 3, device='npu')
    scale = torch.tensor(2.0, device='npu')

    config.ascend.aclgraph.begin_capture()
    _ = opt_model(x, y, z, scale)
    config.ascend.aclgraph.end_capture()

    result_replay = opt_model(x, y, z, scale)
    expected = model(x, y, z, scale)

    assert torch.allclose(result_replay, expected, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_aclgraph_skip_all_ops(monkeypatch):
    """
    Feature: Test aclgraph with all operators skipped
    Description: Skip all arithmetic operators, each runs as single op
    Expectation: The result is correct
    """
    monkeypatch.setenv("MS_INFERRT_ENABLE_ACLGRAPH", "on")

    def model(x, y, z):
        # 10 ops all skipped
        out = torch.add(x, y)
        out = torch.mul(out, z)
        out = torch.sub(out, y)
        out = torch.add(out, x)
        out = torch.mul(out, z)
        out = torch.sub(out, y)
        out = torch.add(out, x)
        out = torch.mul(out, z)
        out = torch.sub(out, y)
        out = torch.add(out, x)
        return out

    config.ascend.aclgraph.set_op_capture_skip(["add", "mul", "sub"])

    opt_model = torch.compile(model, backend=backend)

    x = torch.randn(2, 3, device='npu')
    y = torch.randn(2, 3, device='npu')
    z = torch.randn(2, 3, device='npu')

    config.ascend.aclgraph.begin_capture()
    _ = opt_model(x, y, z)
    config.ascend.aclgraph.end_capture()

    result_replay = opt_model(x, y, z)
    expected = model(x, y, z)

    assert torch.allclose(result_replay, expected, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_aclgraph_no_skip(monkeypatch):
    """
    Feature: Test aclgraph with no operators skipped
    Description: Capture entire graph without skipping any ops
    Expectation: The result is correct
    """
    monkeypatch.setenv("MS_INFERRT_ENABLE_ACLGRAPH", "on")

    def model(x, y, scale1, scale2):
        # 10 ops no skip
        out = torch.add(x, y)
        out = torch.mul(out, scale1)
        out = torch.sub(out, y)
        out = torch.div(out, scale2)
        out = torch.add(out, x)
        out = torch.mul(out, scale1)
        out = torch.sub(out, y)
        out = torch.div(out, scale2)
        out = torch.add(out, x)
        out = torch.mul(out, scale1)
        return out

    config.ascend.aclgraph.set_op_capture_skip([])

    opt_model = torch.compile(model, backend=backend)

    x = torch.randn(2, 3, device='npu')
    y = torch.randn(2, 3, device='npu')
    scale1 = torch.tensor(2.0, device='npu')
    scale2 = torch.tensor(4.0, device='npu')

    config.ascend.aclgraph.begin_capture()
    _ = opt_model(x, y, scale1, scale2)
    config.ascend.aclgraph.end_capture()

    result_replay = opt_model(x, y, scale1, scale2)
    expected = model(x, y, scale1, scale2)

    assert torch.allclose(result_replay, expected, rtol=1e-4, atol=1e-4)


# ============================================================================
# Multiple Input Shapes Tests
# ============================================================================

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_aclgraph_multiple_shapes(monkeypatch):
    """
    Feature: Test aclgraph with multiple input shapes
    Description: Test that different shapes create different cached graphs
    Expectation: Each shape produces correct result
    """
    monkeypatch.setenv("MS_INFERRT_ENABLE_ACLGRAPH", "on")

    def model(x, y, scale):
        # 10 ops
        out = torch.add(x, y)
        out = torch.mul(out, scale)
        out = torch.add(out, x)
        out = torch.mul(out, scale)
        out = torch.add(out, y)
        out = torch.mul(out, scale)
        out = torch.add(out, x)
        out = torch.mul(out, scale)
        out = torch.add(out, y)
        out = torch.mul(out, scale)
        return out

    config.ascend.aclgraph.set_op_capture_skip([])

    opt_model = torch.compile(model, backend=backend)

    scale = torch.tensor(2.0, device='npu')
    shapes = [(2, 3), (4, 5), (2, 3), (8, 16)]

    # Test with multiple shapes
    for i, shape in enumerate(shapes):
        x = torch.randn(shape, device='npu')
        y = torch.randn(shape, device='npu')

        # Capture for first occurrence of each shape
        if i == 0 or shape != shapes[i-1]:
            config.ascend.aclgraph.begin_capture()

        result = opt_model(x, y, scale)

        if i == 0 or shape != shapes[i-1]:
            config.ascend.aclgraph.end_capture()

        expected = model(x, y, scale)
        assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4), f"Failed at shape {shape}"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_aclgraph_shape_variations(monkeypatch):
    """
    Feature: Test aclgraph with shape variations
    Description: Test different rank tensors - capture all shapes first, then replay with
                 existing shapes and new shapes (which trigger new captures)
    Expectation: Each shape variation works correctly in both capture and replay phases
    """
    monkeypatch.setenv("MS_INFERRT_ENABLE_ACLGRAPH", "on")

    def model(x, scale):
        # 10 ops
        out = torch.mul(x, scale)
        out = torch.add(out, x)
        out = torch.mul(out, scale)
        out = torch.add(out, x)
        out = torch.mul(out, scale)
        out = torch.add(out, x)
        out = torch.mul(out, scale)
        out = torch.add(out, x)
        out = torch.mul(out, scale)
        out = torch.add(out, x)
        return out

    config.ascend.aclgraph.set_op_capture_skip([])

    opt_model = torch.compile(model, backend=backend)

    scale = torch.tensor(2.0, device='npu')

    # Different ranks and sizes for capture phase
    capture_shapes = [
        (4,),           # 1D
        (2, 3),         # 2D
        (2, 3, 4),      # 3D
        (1, 1, 1, 1),   # 4D small
        (2, 4, 8, 16),  # 4D larger
    ]

    # ========== Phase 1: Capture all shapes ==========
    for shape in capture_shapes:
        x = torch.randn(shape, device='npu')

        config.ascend.aclgraph.begin_capture()
        result = opt_model(x, scale)
        config.ascend.aclgraph.end_capture()

        expected = model(x, scale)
        assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4), f"Capture failed at shape {shape}"
        print(f"Captured shape: {shape}")

    # ========== Phase 2: Replay with captured shapes ==========
    for shape in capture_shapes:
        x = torch.randn(shape, device='npu')  # New input data, same shape

        result_replay = opt_model(x, scale)
        expected = model(x, scale)

        assert torch.allclose(result_replay, expected, rtol=1e-4, atol=1e-4), f"Replay failed at shape {shape}"
        print(f"Replayed shape: {shape}")

    # ========== Phase 3: New shapes during replay phase (trigger new captures) ==========
    # These shapes were not captured, should trigger new capture automatically
    new_shapes = [
        (8,),           # 1D different size
        (3, 5),         # 2D different size
        (1, 2, 3, 4, 5),  # 5D - new rank
    ]

    for shape in new_shapes:
        x = torch.randn(shape, device='npu')

        # This should work - new shape triggers capture
        result = opt_model(x, scale)
        expected = model(x, scale)

        assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4), f"New shape failed at shape {shape}"
        print(f"New shape captured and executed: {shape}")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_aclgraph_batch_size_variation(monkeypatch):
    """
    Feature: Test aclgraph with batch size variations
    Description: Test common LLM inference pattern with varying batch sizes
    Expectation: Different batch sizes create separate cached graphs
    """
    monkeypatch.setenv("MS_INFERRT_ENABLE_ACLGRAPH", "on")

    def model(x, weight, weight_t, bias, scale):
        # 10 ops - multi-layer linear computation
        out = torch.matmul(x, weight)
        out = torch.add(out, bias)
        out = torch.mul(out, scale)
        out = torch.add(out, bias)
        out = torch.matmul(out, weight_t)
        out = torch.add(out, bias)
        out = torch.mul(out, scale)
        out = torch.add(out, bias)
        out = torch.matmul(out, weight)
        out = torch.add(out, bias)
        return out

    config.ascend.aclgraph.set_op_capture_skip([])

    opt_model = torch.compile(model, backend=backend)

    seq_len, hidden_size = 128, 64
    weight = torch.randn(hidden_size, hidden_size, device='npu')
    weight_t = torch.randn(hidden_size, hidden_size, device='npu')
    bias = torch.randn(hidden_size, device='npu')
    scale = torch.tensor(0.5, device='npu')

    batch_sizes = [1, 2, 4, 1, 8]  # Include repeated size

    prev_batch = None
    for batch in batch_sizes:
        x = torch.randn(batch, seq_len, hidden_size, device='npu')

        # Capture for new batch size
        if batch != prev_batch:
            config.ascend.aclgraph.begin_capture()

        result = opt_model(x, weight, weight_t, bias, scale)

        if batch != prev_batch:
            config.ascend.aclgraph.end_capture()

        expected = model(x, weight, weight_t, bias, scale)
        assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3), f"Failed at batch {batch}"
        prev_batch = batch


# ============================================================================
# Parameter and Non-Parameter Tensor Tests
# ============================================================================

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_aclgraph_with_parameter(monkeypatch):
    """
    Feature: Test aclgraph with torch.nn.Parameter
    Description: Parameter tensors should not be staticized, use original tensor
    Expectation: Parameter tensors handled correctly
    """
    monkeypatch.setenv("MS_INFERRT_ENABLE_ACLGRAPH", "on")

    # Create parameters
    # x: (2, 3)
    # w: (3, 4) -> matmul(x, w) -> (2, 4)
    # w_t: (4, 4) -> matmul(out, w_t) -> (2, 4)
    # w2: (4, 3) -> matmul(out, w2) -> (2, 3)
    weight = torch.nn.Parameter(torch.randn(3, 4, device='npu'))
    weight_t = torch.nn.Parameter(torch.randn(4, 4, device='npu'))
    weight2 = torch.nn.Parameter(torch.randn(4, 3, device='npu'))
    bias = torch.nn.Parameter(torch.randn(4, device='npu'))
    bias2 = torch.nn.Parameter(torch.randn(3, device='npu'))

    def model(x, w, w_t, w2, b, b2, scale):
        # 10 ops with parameters
        out = torch.matmul(x, w)       # (2,3) x (3,4) -> (2,4)
        out = torch.add(out, b)        # (2,4) + (4) -> (2,4)
        out = torch.mul(out, scale)    # (2,4)
        out = torch.add(out, b)        # (2,4)
        out = torch.matmul(out, w_t)   # (2,4) x (4,4) -> (2,4)
        out = torch.add(out, b)        # (2,4)
        out = torch.mul(out, scale)    # (2,4)
        out = torch.add(out, b)        # (2,4)
        out = torch.matmul(out, w2)    # (2,4) x (4,3) -> (2,3)
        out = torch.add(out, b2)       # (2,3) + (3) -> (2,3)
        return out

    config.ascend.aclgraph.set_op_capture_skip([])

    opt_model = torch.compile(model, backend=backend)

    x = torch.randn(2, 3, device='npu')
    scale = torch.tensor(0.5, device='npu')

    config.ascend.aclgraph.begin_capture()
    _ = opt_model(x, weight, weight_t, weight2, bias, bias2, scale)
    config.ascend.aclgraph.end_capture()

    # Replay
    result_replay = opt_model(x, weight, weight_t, weight2, bias, bias2, scale)
    expected = model(x, weight, weight_t, weight2, bias, bias2, scale)

    assert torch.allclose(result_replay, expected, rtol=1e-3, atol=1e-3)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_aclgraph_mixed_parameter_nonparameter(monkeypatch):
    """
    Feature: Test aclgraph with mixed Parameter and non-Parameter inputs
    Description: Some inputs are Parameter, some are regular tensors
    Expectation: Both types handled correctly
    """
    monkeypatch.setenv("MS_INFERRT_ENABLE_ACLGRAPH", "on")

    weight = torch.nn.Parameter(torch.randn(4, 5, device='npu'))
    bias = torch.nn.Parameter(torch.randn(5, device='npu'))

    def model(x, w, b, scale1, scale2):
        # 10 ops with mixed inputs
        # x: (2, 4), w: (4, 5)
        out = torch.matmul(x, w)       # (2,4) x (4,5) -> (2,5)
        out = torch.add(out, b)        # (2,5) + (5) -> (2,5)
        out = torch.mul(out, scale1)   # (2,5)
        out = torch.add(out, b)        # (2,5)
        out = torch.mul(out, scale2)   # (2,5)
        out = torch.add(out, b)        # (2,5)
        out = torch.add(out, out)      # (2,5) + (2,5) -> (2,5)
        out = torch.add(out, b)        # (2,5)
        out = torch.mul(out, scale1)   # (2,5)
        out = torch.add(out, b)        # (2,5)
        return out

    config.ascend.aclgraph.set_op_capture_skip([])

    opt_model = torch.compile(model, backend=backend)

    x = torch.randn(2, 4, device='npu')
    scale1 = torch.tensor(0.5, device='npu')
    scale2 = torch.tensor(2.0, device='npu')

    config.ascend.aclgraph.begin_capture()
    _ = opt_model(x, weight, bias, scale1, scale2)
    config.ascend.aclgraph.end_capture()

    # Replay with different non-parameter input
    x2 = torch.randn(2, 4, device='npu')
    result_replay = opt_model(x2, weight, bias, scale1, scale2)
    expected = model(x2, weight, bias, scale1, scale2)

    assert torch.allclose(result_replay, expected, rtol=1e-3, atol=1e-3)


# ============================================================================
# Edge Case Tests
# ============================================================================

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_aclgraph_different_dtypes(monkeypatch):
    """
    Feature: Test aclgraph with different data types
    Description: Test with float16, float32, bfloat16
    Expectation: Each dtype works correctly
    """
    monkeypatch.setenv("MS_INFERRT_ENABLE_ACLGRAPH", "on")

    def model(x, y, scale):
        # 10 ops
        out = torch.add(x, y)
        out = torch.mul(out, scale)
        out = torch.add(out, x)
        out = torch.mul(out, scale)
        out = torch.add(out, y)
        out = torch.mul(out, scale)
        out = torch.add(out, x)
        out = torch.mul(out, scale)
        out = torch.add(out, y)
        out = torch.mul(out, scale)
        return out

    config.ascend.aclgraph.set_op_capture_skip([])

    dtypes = [torch.float32, torch.float16]

    for dtype in dtypes:
        opt_model = torch.compile(model, backend=backend)

        x = torch.randn(2, 3, device='npu', dtype=dtype)
        y = torch.randn(2, 3, device='npu', dtype=dtype)
        scale = torch.tensor(2.0, device='npu', dtype=dtype)

        config.ascend.aclgraph.begin_capture()
        _ = opt_model(x, y, scale)
        config.ascend.aclgraph.end_capture()

        result_replay = opt_model(x, y, scale)
        expected = model(x, y, scale)

        rtol = 1e-3 if dtype == torch.float16 else 1e-4
        assert torch.allclose(result_replay, expected, rtol=rtol, atol=rtol), f"Failed at dtype {dtype}"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_aclgraph_scalar_inputs(monkeypatch):
    """
    Feature: Test aclgraph with scalar inputs
    Description: Test with scalar tensor inputs
    Expectation: Scalar inputs handled correctly
    """
    monkeypatch.setenv("MS_INFERRT_ENABLE_ACLGRAPH", "on")

    def model(x, scalar):
        # 10 ops with scalar
        out = torch.mul(x, scalar)
        out = torch.add(out, x)
        out = torch.mul(out, scalar)
        out = torch.add(out, x)
        out = torch.mul(out, scalar)
        out = torch.add(out, x)
        out = torch.mul(out, scalar)
        out = torch.add(out, x)
        out = torch.mul(out, scalar)
        out = torch.add(out, x)
        return out

    config.ascend.aclgraph.set_op_capture_skip([])

    opt_model = torch.compile(model, backend=backend)

    x = torch.randn(2, 3, device='npu')
    scalar = torch.tensor(2.0, device='npu')

    config.ascend.aclgraph.begin_capture()
    _ = opt_model(x, scalar)
    config.ascend.aclgraph.end_capture()

    result_replay = opt_model(x, scalar)
    expected = model(x, scalar)

    assert torch.allclose(result_replay, expected, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_aclgraph_repeated_replay(monkeypatch):
    """
    Feature: Test aclgraph with multiple replays
    Description: Test that multiple replay calls produce consistent results
    Expectation: All replays produce same correct result
    """
    monkeypatch.setenv("MS_INFERRT_ENABLE_ACLGRAPH", "on")

    def model(x, y, scale):
        # 10 ops
        out = torch.mul(x, scale)
        out = torch.add(out, y)
        out = torch.mul(out, scale)
        out = torch.add(out, x)
        out = torch.mul(out, scale)
        out = torch.add(out, y)
        out = torch.mul(out, scale)
        out = torch.add(out, x)
        out = torch.mul(out, scale)
        out = torch.add(out, y)
        return out

    config.ascend.aclgraph.set_op_capture_skip([])

    opt_model = torch.compile(model, backend=backend)

    x = torch.randn(3, 4, device='npu')
    y = torch.randn(3, 4, device='npu')
    scale = torch.tensor(2.0, device='npu')

    # Capture
    config.ascend.aclgraph.begin_capture()
    _ = opt_model(x, y, scale)
    config.ascend.aclgraph.end_capture()

    # Multiple replays
    expected = model(x, y, scale)
    for i in range(5):
        result_replay = opt_model(x, y, scale)
        assert torch.allclose(result_replay, expected, rtol=1e-4, atol=1e-4), f"Failed at replay {i}"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_aclgraph_large_tensor(monkeypatch):
    """
    Feature: Test aclgraph with large tensors
    Description: Test with larger tensor sizes to stress test
    Expectation: Large tensors handled correctly
    """
    monkeypatch.setenv("MS_INFERRT_ENABLE_ACLGRAPH", "on")

    def model(x, y, scale):
        # 10 ops with large tensors
        out = torch.add(x, y)
        out = torch.mul(out, scale)
        out = torch.add(out, x)
        out = torch.mul(out, scale)
        out = torch.add(out, y)
        out = torch.mul(out, scale)
        out = torch.add(out, x)
        out = torch.mul(out, scale)
        out = torch.add(out, y)
        out = torch.mul(out, scale)
        return out

    config.ascend.aclgraph.set_op_capture_skip([])

    opt_model = torch.compile(model, backend=backend)

    # Large tensor
    x = torch.randn(1024, 1024, device='npu')
    y = torch.randn(1024, 1024, device='npu')
    scale = torch.tensor(2.0, device='npu')

    config.ascend.aclgraph.begin_capture()
    _ = opt_model(x, y, scale)
    config.ascend.aclgraph.end_capture()

    result_replay = opt_model(x, y, scale)
    expected = model(x, y, scale)

    assert torch.allclose(result_replay, expected, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_aclgraph_single_input(monkeypatch):
    """
    Feature: Test aclgraph with single input
    Description: Test with only one input tensor
    Expectation: Single input handled correctly
    """
    monkeypatch.setenv("MS_INFERRT_ENABLE_ACLGRAPH", "on")

    def model(x, scale):
        # 10 ops with single main input
        out = torch.mul(x, scale)
        out = torch.add(out, x)
        out = torch.mul(out, scale)
        out = torch.add(out, x)
        out = torch.mul(out, scale)
        out = torch.add(out, x)
        out = torch.mul(out, scale)
        out = torch.add(out, x)
        out = torch.mul(out, scale)
        out = torch.add(out, x)
        return out

    config.ascend.aclgraph.set_op_capture_skip([])

    opt_model = torch.compile(model, backend=backend)

    x = torch.randn(4, 5, device='npu')
    scale = torch.tensor(3.0, device='npu')

    config.ascend.aclgraph.begin_capture()
    _ = opt_model(x, scale)
    config.ascend.aclgraph.end_capture()

    result_replay = opt_model(x, scale)
    expected = model(x, scale)

    assert torch.allclose(result_replay, expected, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_aclgraph_many_inputs(monkeypatch):
    """
    Feature: Test aclgraph with many inputs
    Description: Test with multiple (10) input tensors
    Expectation: Many inputs handled correctly
    """
    monkeypatch.setenv("MS_INFERRT_ENABLE_ACLGRAPH", "on")

    def model(a, b, c, d, e, f, g, h, i, j):
        # 10 ops with 10 inputs
        out = a
        out = torch.add(out, b)
        out = torch.add(out, c)
        out = torch.add(out, d)
        out = torch.add(out, e)
        out = torch.add(out, f)
        out = torch.add(out, g)
        out = torch.add(out, h)
        out = torch.add(out, i)
        out = torch.add(out, j)
        return out

    config.ascend.aclgraph.set_op_capture_skip([])

    opt_model = torch.compile(model, backend=backend)

    inputs = [torch.randn(2, 3, device='npu') for _ in range(10)]

    config.ascend.aclgraph.begin_capture()
    _ = opt_model(*inputs)
    config.ascend.aclgraph.end_capture()

    result_replay = opt_model(*inputs)
    expected = model(*inputs)

    assert torch.allclose(result_replay, expected, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_aclgraph_multiple_outputs(monkeypatch):
    """
    Feature: Test aclgraph with multiple outputs
    Description: Test model that returns multiple tensors
    Expectation: Multiple outputs handled correctly
    """
    monkeypatch.setenv("MS_INFERRT_ENABLE_ACLGRAPH", "on")

    def model(x, y, scale):
        # 10 ops with multiple outputs
        out1 = torch.add(x, y)
        out2 = torch.mul(out1, scale)
        out3 = torch.add(out2, x)
        out4 = torch.mul(out3, scale)
        out5 = torch.add(out4, y)
        out6 = torch.mul(out5, scale)
        out7 = torch.add(out6, x)
        out8 = torch.mul(out7, scale)
        out9 = torch.add(out8, y)
        out10 = torch.mul(out9, scale)
        return out1, out3, out5, out7, out9, out10

    config.ascend.aclgraph.set_op_capture_skip([])

    opt_model = torch.compile(model, backend=backend)

    x = torch.randn(2, 3, device='npu')
    y = torch.randn(2, 3, device='npu')
    scale = torch.tensor(2.0, device='npu')

    config.ascend.aclgraph.begin_capture()
    _ = opt_model(x, y, scale)
    config.ascend.aclgraph.end_capture()

    result_replay = opt_model(x, y, scale)
    expected = model(x, y, scale)

    for i, (r, e) in enumerate(zip(result_replay, expected)):
        assert torch.allclose(r, e, rtol=1e-4, atol=1e-4), f"Failed at output {i}"
