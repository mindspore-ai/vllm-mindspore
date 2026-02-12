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
"""Tests for custom torch operations."""
import pytest
import torch
import torch_npu

from mrt.torch.fx_backend import backend as fx_backend
from mrt.torch.fx_mlir_backend import backend as mlir_backend
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend, mlir_backend))
def test_aten_ldexp(backend):
    """
    Feature: Check aten ldexp op launch
    Description: Check aten ldexp op launch
    Expectation: The result is correct
    """

    def aten_ldexp(x, y):
        return torch.ops.aten.ldexp(x, y)

    ldexp_compiled = torch.compile(aten_ldexp, backend=backend)
    x = torch.randn([2, 16], dtype=torch.float32).npu()
    y = torch.randn([2, 16], dtype=torch.float32).npu()
    z = ldexp_compiled(x, y)
    expected = torch.ops.aten.ldexp(x, y)
    assert torch.allclose(z, expected)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend, mlir_backend))
def test_npu_quant_matmul_with_nz_format(backend):
    """
    Feature: Check npu_quant_matmul op launch
    Description: Check npu_quant_matmul op launch
    Expectation: The result is correct
    """

    def npu_quant_matmul_func(x, y, scale, output_dtype):
        return torch_npu.npu_quant_matmul(x, y, scale, output_dtype=output_dtype)

    npu_quant_matmul_compiled = torch.compile(npu_quant_matmul_func, backend=backend)
    x = torch.randint(-5, 5, (2, 16), dtype=torch.int8).npu()
    y = torch.randint(-5, 5, (16, 32), dtype=torch.int8).npu()
    scale = torch.randn(1, dtype=torch.float32).npu()
    y_nz = torch_npu.npu_format_cast(y, 29).npu()
    z = npu_quant_matmul_compiled(x, y_nz, scale, output_dtype=torch.bfloat16)
    expected = torch_npu.npu_quant_matmul(x, y_nz, scale, output_dtype=torch.bfloat16)
    assert torch.allclose(z, expected)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend, mlir_backend))
def test_aten_ldexp_cache_hit(backend):
    """
    Feature: Check aten ldexp op with cache hit scenario
    Description: Verify correctness when backend input/output conversion cache is hit
    Expectation: The result is correct when reusing cached conversion for same input shapes
    """

    def aten_ldexp(x, y):
        return torch.ops.aten.ldexp(x, y)

    ldexp_compiled = torch.compile(aten_ldexp, backend=backend)

    # First call: establish cache
    x1 = torch.randn([2, 16], dtype=torch.float32).npu()
    y1 = torch.randn([2, 16], dtype=torch.float32).npu()
    z1 = ldexp_compiled(x1, y1)

    # Second call: same shape and dtype, triggers cache hit
    x2 = torch.randn([2, 16], dtype=torch.float32).npu()
    y2 = torch.randn([2, 16], dtype=torch.float32).npu()
    z2 = ldexp_compiled(x2, y2)

    # Verify both results are correct (compared with eager mode)
    expected1 = torch.ops.aten.ldexp(x1, y1)
    expected2 = torch.ops.aten.ldexp(x2, y2)

    assert torch.allclose(z1, expected1), "First call result mismatch"
    assert torch.allclose(z2, expected2), "Cache hit call result mismatch"


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend, mlir_backend))
def test_aten_ldexp_cache_miss(backend):
    """
    Feature: Check aten ldexp op with cache miss scenario
    Description: Verify correctness when backend input/output conversion cache misses due to different shapes
    Expectation: The result is correct when cache misses and new conversion is performed
    """

    def aten_ldexp(x, y):
        return torch.ops.aten.ldexp(x, y)

    ldexp_compiled = torch.compile(aten_ldexp, backend=backend)

    # First call: establish cache for shape [2, 16]
    x1 = torch.randn([2, 16], dtype=torch.float32).npu()
    y1 = torch.randn([2, 16], dtype=torch.float32).npu()
    z1 = ldexp_compiled(x1, y1)

    # Second call: different shape, triggers cache miss
    x2 = torch.randn([4, 32], dtype=torch.float32).npu()
    y2 = torch.randn([4, 32], dtype=torch.float32).npu()
    z2 = ldexp_compiled(x2, y2)

    # Third call: new shape again, triggers another cache miss
    x3 = torch.randn([8, 8], dtype=torch.float32).npu()
    y3 = torch.randn([8, 8], dtype=torch.float32).npu()
    z3 = ldexp_compiled(x3, y3)

    # Verify all results are correct
    expected1 = torch.ops.aten.ldexp(x1, y1)
    expected2 = torch.ops.aten.ldexp(x2, y2)
    expected3 = torch.ops.aten.ldexp(x3, y3)

    assert torch.allclose(z1, expected1), "First call result mismatch"
    assert torch.allclose(z2, expected2), "Cache miss call (shape [4,32]) result mismatch"
    assert torch.allclose(z3, expected3), "Cache miss call (shape [8,8]) result mismatch"
