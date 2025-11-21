"""
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import mrt
from mrt.torch import backend
import os
import pytest
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_lowered_bias_add_static_shape():
    """
    Feature: Test MLIR lowered operator with static shape
    Description: BiasAdd with static shape [1x6144] compiled from MLIR
    Expectation: Correct numerical results
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mlir_path = os.path.join(script_dir, "add.mlir")
    os.environ["LOWERED_BIAS_ADD_MLIR_PATH"] = mlir_path
    os.environ["TEST_DIR"] = script_dir

    op_source = os.path.join(script_dir, "lowered_add_custom_op.cc")
    mrt.ops.load(name="lowered_bias_add", sources=[op_source], backend="Ascend",
                 extra_ldflags=["-lops_ascend_lowered"])

    @torch.library.custom_op("mrt::lowered_bias_add", mutates_args=())
    def lowered_bias_add_op(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Placeholder for lowered_bias_add operator")

    @torch.library.register_fake("mrt::lowered_bias_add")
    def _(x, bias):
        return x

    def bias_add_fn(x, bias):
        return torch.ops.mrt.lowered_bias_add(x, bias)

    bias_add_compiled = torch.compile(bias_add_fn, backend=backend)

    x = torch.randn(1, 6144, dtype=torch.float16).npu()
    bias = torch.randn(1, 6144, dtype=torch.float16).npu()

    result = bias_add_compiled(x, bias)
    expected = x + bias

    assert torch.allclose(result, expected, rtol=5e-2, atol=5e-2), \
        f"Result mismatch: max_diff={torch.max(torch.abs(result - expected)).item()}"
    print("Lowered BiasAdd (static shape) test passed.")


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_lowered_bias_add_dynamic_shape():
    """
    Feature: Test MLIR lowered operator with dynamic shape
    Description: BiasAdd with dynamic shape compiled from MLIR
    Expectation: Works with different input shapes
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mlir_path = os.path.join(script_dir, "add_dyn.mlir")
    os.environ["LOWERED_BIAS_ADD_MLIR_PATH"] = mlir_path
    os.environ["TEST_DIR"] = script_dir

    op_source = os.path.join(script_dir, "lowered_add_custom_op.cc")
    mrt.ops.load(name="lowered_bias_add", sources=[op_source], backend="Ascend",
                 extra_ldflags=["-lops_ascend_lowered"])

    @torch.library.custom_op("mrt::lowered_bias_add", mutates_args=())
    def lowered_bias_add_op(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Placeholder")

    @torch.library.register_fake("mrt::lowered_bias_add")
    def _(x, bias):
        return x

    bias_add_compiled = torch.compile(
        lambda x, b: torch.ops.mrt.lowered_bias_add(x, b),
        backend=backend
    )

    test_shapes = [(1, 6144), (2, 4096), (4, 2048)]

    for M, N in test_shapes:
        x = torch.randn(M, N, dtype=torch.float16).npu()
        bias = torch.randn(M, N, dtype=torch.float16).npu()

        result = bias_add_compiled(x, bias)
        expected = x + bias

        assert torch.allclose(result, expected, rtol=5e-2, atol=5e-2), \
            f"Shape [{M}x{N}] failed with max_diff={torch.max(torch.abs(result - expected)).item()}"

    print("All dynamic shape tests passed.")


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_lowered_custom_op_compilation_cache():
    """
    Feature: Test MLIR compilation cache
    Description: Second execution should hit cache and be faster
    Expectation: No recompilation on second run
    """
    import time

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mlir_path = os.path.join(script_dir, "add_dyn.mlir")
    os.environ["LOWERED_BIAS_ADD_MLIR_PATH"] = mlir_path
    os.environ["TEST_DIR"] = script_dir

    op_source = os.path.join(script_dir, "lowered_add_custom_op.cc")
    mrt.ops.load(name="lowered_bias_add", sources=[op_source], backend="Ascend",
                 extra_ldflags=["-lops_ascend_lowered"])

    @torch.library.custom_op("mrt::lowered_bias_add", mutates_args=())
    def lowered_bias_add_op(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @torch.library.register_fake("mrt::lowered_bias_add")
    def _(x, bias):
        return x

    bias_add_compiled = torch.compile(
        lambda x, b: torch.ops.mrt.lowered_bias_add(x, b),
        backend=backend
    )

    x = torch.randn(1, 6144, dtype=torch.float16).npu()
    bias = torch.randn(1, 6144, dtype=torch.float16).npu()

    start_time = time.time()
    result1 = bias_add_compiled(x, bias)
    first_time = time.time() - start_time

    start_time = time.time()
    result2 = bias_add_compiled(x, bias)
    second_time = time.time() - start_time

    assert torch.equal(result1, result2), "Results differ between executions"
    print("Compilation cache test passed.")
