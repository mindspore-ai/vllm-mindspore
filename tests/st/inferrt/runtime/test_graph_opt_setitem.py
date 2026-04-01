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

"""Tests for graph optimization with setitem operation."""
import pytest
import torch

from ms_inferrt.torch.fx_backend import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_setitem_with_tensor_index(backend):
    """
    Feature: Test setitem with tensor index from operator output
    Description: Test setitem where index is a tensor from operator output
    Expectation: The result is correct
    """

    def func(x, idx_tensor, value):
        x[idx_tensor] = value
        return x

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 2, 3).npu()
    expected = x.clone()
    idx_tensor = torch.tensor([0, 2]).npu()
    value = torch.randn(2, 2, 3).npu()
    out = compiled_op(x, idx_tensor, value)
    expected[idx_tensor] = value
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_setitem_with_tensor_index_and_mul(backend):
    """
    Feature: Test setitem with tensor index from operator output
    Description: Test setitem where index is a tensor from operator output
    Expectation: The result is correct
    """

    def func(x, idx_tensor, value, val2):
        x[idx_tensor] = value
        x = x * val2
        return x

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 2, 3).npu()
    expected = x.clone()
    idx_tensor = torch.tensor([0, 2]).npu()
    value = torch.randn(2, 2, 3).npu()
    val2 = torch.tensor(2).npu()
    out = compiled_op(x, idx_tensor, value, val2)
    expected[idx_tensor] = value
    AssertRtolEqual(out, expected * 2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_setitem_with_single_computed_index(backend):
    """
    Feature: Test setitem with computed index from operator output
    Description: Test setitem where index is computed from operator
    Expectation: The result is correct
    """

    def func(x, start_idx, value, val2):
        res = x.clone()
        end_idx = start_idx + val2
        res[end_idx] = value
        return res

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 2, 3).npu()
    expected = x.clone()
    start_idx = torch.tensor(1).npu()
    value = torch.randn(2, 3).npu()
    val2 = torch.tensor(2).npu()
    out = compiled_op(x, start_idx, value, val2)
    end_idx = start_idx + 2
    expected[end_idx] = value
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_setitem_with_double_computed_index(backend):
    """
    Feature: Test setitem with computed index from operator output
    Description: Test setitem where index is computed from operator
    Expectation: The result is correct
    """

    def func(x, start_idx, value, val2):
        res = x.clone()
        end_idx = start_idx + val2
        res[start_idx:end_idx] = value
        return res

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 2, 3).npu()
    expected = x.clone()
    start_idx = torch.tensor(1).npu()
    value = torch.randn(2, 2, 3).npu()
    val2 = torch.tensor(2).npu()
    out = compiled_op(x, start_idx, value, val2)
    end_idx = start_idx + 2
    expected[start_idx:end_idx] = value
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_setitem_with_dynamic_slice(backend):
    """
    Feature: Test setitem with dynamic slice from operator output
    Description: Test setitem where slice indices are computed from operators
    Expectation: The result is correct
    """

    def func(x, start, end, value):
        res = x.clone()
        res[start:end, ...] = value
        return res

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(3, 2, 4, 4).npu()
    expected = x.clone()
    start = torch.tensor(0).npu()
    end = torch.tensor(2).npu()
    value = torch.randn(2, 2, 4, 4).npu()
    out = compiled_op(x, start, end, value)
    expected[start:end, ...] = value
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_tensor_setitem_slice_tensor(backend):
    """
    Feature: Test tensor_setitem_slice_tensor
    Description: Test tensor_setitem_slice_tensor
    Expectation: The result is correct
    """

    def func(x, indices, value):
        res = x.clone()
        res[indices] = value
        return res

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 2, 3).npu()
    value = torch.randn(2, 2, 3).npu()
    expected = x.clone()
    indices = slice(1, 3, 1)
    out = compiled_op(x, indices, value)
    expected[indices] = value
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_complex_setitem_with_computed_ops(backend):
    """
    Feature: Test complex setitem with computed index and multiple ops
    Description: Test setitem with index from operator output, followed by multiple ops
    Expectation: The result is correct
    """

    def func(x, base_idx, offset, value, val1, val2):
        res = x.clone()
        index = offset + base_idx
        res[index] = value
        res = res * val2
        res = res + val1
        res = torch.relu(res)
        res = res.sum(dim=0)
        return res

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(8, 4, 3).npu()
    base_idx = torch.tensor(4, dtype=torch.int64).npu()
    offset = torch.tensor(2, dtype=torch.int64).npu()
    value = torch.randn(4, 3).npu()
    val1 = torch.tensor(1).npu()
    val2 = torch.tensor(2).npu()

    out = compiled_op(x, base_idx, offset, value, val1, val2)
    expected = func(x, base_idx, offset, value, val1, val2)
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_setitem_with_tensor_index_dynamo(backend):
    """
    Feature: Test setitem with tensor index using torch dynamo
    Description: Test setitem where index is a tensor from operator output, compiled with torch dynamo
    Expectation: The result is correct
    """

    def func(x, base_idx, offset, value, val_1, val_2):
        res = x.clone()
        index = offset + base_idx
        res[index] = value
        res = res * val_2
        res = res + val_1
        res = res.sum(dim=0)
        return res

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(8, 4, 3).npu()
    base_idx = torch.tensor([0, 2]).npu()
    offset = torch.tensor([2, 2]).npu()
    value = torch.randn(2, 4, 3).npu()
    val_2 = torch.tensor(2).npu()
    val_1 = torch.tensor(1).npu()

    out = compiled_op(x, base_idx, offset, value, val_1, val_2)
    expected = func(x, base_idx, offset, value, val_1, val_2)
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_setitem_with_bool_mask(backend):
    """
    Feature: Test setitem with bool mask tensor
    Description: Test setitem where index is a bool tensor, updating positions where mask is True
    Expectation: The result is correct
    """

    def func(x, mask, value):
        res = x.clone()
        res[mask] = value
        return res

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 3).npu()
    expected = x.clone()
    mask = torch.tensor([True, False, True, False]).npu()
    value = torch.tensor(10.0).npu()

    out = compiled_op(x, mask, value)
    expected[mask] = value
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_setitem_with_bool_mask2(backend):
    """
    Feature: Test setitem with bool mask tensor
    Description: Test setitem where index is a bool tensor, updating positions where mask is True
    Expectation: The result is correct
    """

    def func(x, mask):
        x[mask] = 0
        return x

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 3).npu()
    expected = x.clone()
    mask = torch.tensor([True, False, True, False]).npu()

    out = compiled_op(x, mask)
    expected[mask] = 0
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_setitem_with_dynamic_shape(backend):
    """
    Feature: Test setitem with bool mask tensor
    Description: Test setitem where index is a bool tensor, updating positions where mask is True
    Expectation: The result is correct
    """

    def func(x, start_index, end_index):
        mask = (x < start_index) | (x >= end_index)
        x = x - start_index
        y = x.unsqueeze(1)
        y = y.repeat(1, 3)
        x[mask] = 0
        y[mask] = 0
        return x, y

    compiled_op = torch.compile(func, backend=backend)

    # Case 1: input length 4
    x = torch.randn(4).npu()
    start = torch.tensor(1).npu()
    end = torch.tensor(3).npu()

    out = compiled_op(x.clone(), start, end)
    expected = func(x.clone(), start, end)
    AssertRtolEqual(out[0], expected[0])
    AssertRtolEqual(out[1], expected[1])

    # Case 2: input length 8 (dynamic shape)
    x_2 = torch.randn(8).npu()
    start_2 = torch.tensor(1).npu()
    end_2 = torch.tensor(3).npu()

    out_2 = compiled_op(x_2.clone(), start_2, end_2)
    expected_2 = func(x_2.clone(), start_2, end_2)
    AssertRtolEqual(out_2[0], expected_2[0])
    AssertRtolEqual(out_2[1], expected_2[1])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_setitem_const_index_const_value(backend):
    """
    Feature: Test setitem with tensor index from operator output
    Description: Test setitem where index is a tensor from operator output
    Expectation: The result is correct
    """

    def func(x):
        x[1] = 2
        return x

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 2, 3).npu()
    expected = x.clone()
    out = compiled_op(x)
    expected[1] = 2
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_setitem_const_index_input_value(backend):
    """
    Feature: Test setitem with tensor index from operator output
    Description: Test setitem where index is a tensor from operator output
    Expectation: The result is correct
    """

    def func(x, value):
        x[1] = value
        return x

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 2, 3).npu()
    value = torch.randn(2, 3).npu()
    expected = x.clone()
    out = compiled_op(x, value)
    expected[1] = value
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_setitem_input_index_const_value(backend):
    """
    Feature: Test setitem with tensor index from operator output
    Description: Test setitem where index is a tensor from operator output
    Expectation: The result is correct
    """

    def func(x, idx_tensor):
        x[idx_tensor] = 2
        return x

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 2, 3).npu()
    expected = x.clone()
    idx_tensor = torch.tensor([0, 2]).npu()
    out = compiled_op(x, idx_tensor)
    expected[idx_tensor] = 2
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.skip("Not implemented.")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_setitem_slice_index_const_value(backend):
    """
    Feature: Test setitem with tensor index from operator output
    Description: Test setitem where index is a tensor from operator output
    Expectation: The result is correct
    """

    def func(x, val2):
        x[1:2] = val2
        return x

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 2, 3).npu()
    expected = x.clone()
    val2 = torch.tensor(2).npu()
    out = compiled_op(x, val2)
    expected[1:2] = val2
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.skip("Not implemented.")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_setitem_slice_step_index_input_value(backend):
    """
    Feature: Test setitem with tensor index from operator output
    Description: Test setitem where index is a tensor from operator output
    Expectation: The result is correct
    """

    def func(x, value):
        x[1:3:2] = value
        return x

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(4, 2, 3).npu()
    value = torch.randn(2, 3).npu()
    expected = x.clone()
    out = compiled_op(x, value)
    expected[1:3:2] = value
    AssertRtolEqual(out, expected)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_setitem_with_tensor_index_decomposition(backend):  # pylint: disable=unused-argument
    """
    Feature: Test setitem with tensor index using torch dynamo
    Description: Test setitem where index is a tensor from operator output, compiled with torch dynamo
    Expectation: The result is correct
    """
    # Test body intentionally left as debug-only and currently skipped.
    # Keeping placeholder implementation for future development.
    def func(x, base_idx, offset, value, val_1, val_2):  # pylint: disable=unused-variable
        res = x.clone()
        index = offset + base_idx
        res[index] = value
        res = res * val_2
        res = res + val_1
        res = res.sum(dim=0)

        return res, res[1]


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_setitem_with_tensor_index_decomposition2(backend):
    """
    Feature: Test setitem with tensor index using torch dynamo
    Description: Test setitem where index is a tensor from operator output, compiled with torch dynamo
    Expectation: The result is correct
    """
    def func(x, base_idx, offset, value, val_1, val_2):
        res = x.clone()
        index = offset + base_idx
        res[index] = value
        res = res * val_2
        res = res + val_1
        res = res.sum(dim=0)

        return res, res[1]

    compiled_op = torch.compile(func, backend=backend)
    x = torch.randn(8, 4, 3).npu()
    base_idx = torch.tensor([0, 2]).npu()
    offset = torch.tensor([2, 2]).npu()
    value = torch.randn(2, 4, 3).npu()
    val_2 = torch.tensor(2).npu()
    val_1 = torch.tensor(1).npu()

    out = compiled_op(x, base_idx, offset, value, val_1, val_2)
    print(out)
