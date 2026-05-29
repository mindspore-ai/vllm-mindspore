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
"""Test aclnn compare operators."""

import pytest
import torch

from ms_inferrt.torch import fx_mlir_backend as backend
from ms_inferrt.torch.fx_backend import backend as fx_backend
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("ops", ("eq", "ne", "lt", "le", "gt", "ge"))
@pytest.mark.parametrize("datatype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
def test_compare_ops_float(ops, datatype):
    """
    Feature: Check aclnn compare op launch
    Description: Check aclnn op launch with cache
    Expectation: The result is correct
    """
    def compare_ops(x, y):
        exec_op = getattr(torch, ops)
        return exec_op(x, y)

    compare_model = torch.compile(compare_ops, backend=backend)
    shapes = ((2, 2), (100, 100))

    for shape in shapes:
        x = torch.rand(shape, device="npu", dtype=datatype)
        y = torch.rand(shape, device="npu", dtype=datatype)

        x_cpu = x.cpu()
        y_cpu = y.cpu()

        expected = compare_ops(x_cpu, y_cpu)
        cache_miss_actual = compare_model(x, y)
        cache_hit_actual = compare_model(x, y)

        assert torch.equal(cache_miss_actual.cpu(), expected), \
            f"\ncache_miss_actual={cache_hit_actual}\nexpected={expected}"
        assert torch.equal(cache_hit_actual.cpu(), expected), \
            f"\ncache_hit_actual={cache_hit_actual}\nexpected={expected}"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("ops", ("eq", "ne", "lt", "le", "gt", "ge"))
@pytest.mark.parametrize("datatype", (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64))
def test_compare_ops_int(ops, datatype):
    """
    Feature: Check aclnn compare op launch
    Description: Check aclnn op launch with cache
    Expectation: The result is correct
    """
    def compare_ops(x, y):
        exec_op = getattr(torch, ops)
        return exec_op(x, y)

    compare_model = torch.compile(compare_ops, backend=backend)
    shapes = ((10, 10), (100, 100))

    for shape in shapes:
        x = torch.randint(0, 100, shape, device="npu", dtype=datatype)
        y = torch.randint(0, 100, shape, device="npu", dtype=datatype)

        x_cpu = x.cpu()
        y_cpu = y.cpu()

        expected = compare_ops(x_cpu, y_cpu)
        cache_miss_actual = compare_model(x, y)
        cache_hit_actual = compare_model(x, y)

        assert torch.equal(cache_miss_actual.cpu(), expected), \
            f"\ncache_miss_actual={cache_hit_actual}\nexpected={expected}"
        assert torch.equal(cache_hit_actual.cpu(), expected), \
            f"\ncache_hit_actual={cache_hit_actual}\nexpected={expected}"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("ops", ("eq", "ne", "lt", "le", "gt", "ge"))
@pytest.mark.parametrize("datatype", (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64))
def test_compare_ops_float_broadcast(ops, datatype):
    """
    Feature: Check aclnn compare op launch
    Description: Check aclnn op launch with cache
    Expectation: The result is correct
    """
    def compare_ops(x, y):
        exec_op = getattr(torch, ops)
        return exec_op(x, y)

    compare_model = torch.compile(compare_ops, backend=backend)
    shapes = (((1,), (2,)),
              ((2, 1), (2,)),
              ((1, 2), (2,)),
              ((3, 2), (2,)),
              ((1, 3, 2), (2,)),
              ((1, 3, 2), (3, 2)),
              ((3, 1, 2), (3, 2)),
              ((3, 1, 2), (1, 3, 2)),
              )

    for shape in shapes:
        shape_x, shape_y = shape

        x = torch.rand(shape_x, device="npu", dtype=datatype)
        y = torch.rand(shape_y, device="npu", dtype=datatype)

        x_cpu = x.cpu()
        y_cpu = y.cpu()

        expected = compare_ops(x_cpu, y_cpu)
        cache_miss_actual = compare_model(x, y)
        cache_hit_actual = compare_model(x, y)

        assert torch.equal(cache_miss_actual.cpu(), expected), \
            f"\ncache_miss_actual={cache_hit_actual}\nexpected={expected}"
        assert torch.equal(cache_hit_actual.cpu(), expected), \
            f"\ncache_hit_actual={cache_hit_actual}\nexpected={expected}"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("ops", ("eq", "ne", "lt", "le", "gt", "ge"))
@pytest.mark.parametrize("datatype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
def test_compare_ops_int_broadcast(ops, datatype):
    """
    Feature: Check aclnn compare op launch
    Description: Check aclnn op launch with cache
    Expectation: The result is correct
    """
    def compare_ops(x, y):
        exec_op = getattr(torch, ops)
        return exec_op(x, y)

    compare_model = torch.compile(compare_ops, backend=backend)
    shapes = (((1,), (2,)),
              ((2, 1), (2,)),
              ((1, 2), (2,)),
              ((3, 2), (2,)),
              ((1, 3, 2), (2,)),
              ((1, 3, 2), (3, 2)),
              ((3, 1, 2), (3, 2)),
              ((3, 1, 2), (1, 3, 2)),
              )

    for shape in shapes:
        shape_x, shape_y = shape

        x = torch.randint(0, 100, shape_x, device="npu", dtype=datatype)
        y = torch.randint(0, 100, shape_y, device="npu", dtype=datatype)

        x_cpu = x.cpu()
        y_cpu = y.cpu()

        expected = compare_ops(x_cpu, y_cpu)
        cache_miss_actual = compare_model(x, y)
        cache_hit_actual = compare_model(x, y)

        assert torch.equal(cache_miss_actual.cpu(), expected), \
            f"\ncache_miss_actual={cache_hit_actual}\nexpected={expected}"
        assert torch.equal(cache_hit_actual.cpu(), expected), \
            f"\ncache_hit_actual={cache_hit_actual}\nexpected={expected}"


def compare_via_magic_eq(x, y):
    """Use x == y to trigger call_method[target=__eq__] in FX graph, testing the __eq__ mapping in fx_backend."""
    return x == y


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("datatype", (torch.float32, torch.int32))
def test_magic_eq(datatype):
    """
    Feature: Test __eq__ mapping via call_method
    Description: x == y triggers __eq__ string key in _OP_MAP, ensuring no custom_call fallback
    Expectation: Result matches eager, covers problem 3 (__eq__ mapping)
    """
    compile_op = torch.compile(compare_via_magic_eq, backend=fx_backend)
    shapes = ((2, 8), (16, 4))

    for shape in shapes:
        if datatype.is_floating_point:
            x = torch.rand(shape, device="npu", dtype=datatype)
            y = torch.rand(shape, device="npu", dtype=datatype)
        else:
            x = torch.randint(0, 10, shape, device="npu", dtype=datatype)
            y = torch.randint(0, 10, shape, device="npu", dtype=datatype)

        x_cpu = x.cpu()
        y_cpu = y.cpu()

        expected = compare_via_magic_eq(x_cpu, y_cpu)
        actual = compile_op(x, y)

        assert torch.equal(actual.cpu(), expected), f"fx_backend __eq__ mismatch for {datatype}"
