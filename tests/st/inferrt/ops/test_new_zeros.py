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
"""Tests for torch.Tensor.new_zeros operation."""
import pytest
import torch

from ms_inferrt.torch.fx_backend import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def _new_zeros_eager(self_tensor, new_shape, size_format, dtype=None):
    """Eager reference on NPU."""
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if size_format == "tuple":
        size = [] if len(new_shape) == 0 else list(new_shape)
        return self_tensor.new_zeros(size, **kwargs)
    return self_tensor.new_zeros(*new_shape, **kwargs)


def _get_new_zeros_compiled(size_format):
    """new_zeros via fx_backend."""

    if size_format == "tuple":

        def custom_op_func(self_tensor, new_shape, dtype=None):
            kwargs = {}
            if dtype is not None:
                kwargs["dtype"] = dtype
            size = [] if len(new_shape) == 0 else new_shape
            return self_tensor.new_zeros(size, **kwargs)

    else:

        def custom_op_func(self_tensor, new_shape, dtype=None):
            kwargs = {}
            if dtype is not None:
                kwargs["dtype"] = dtype
            return self_tensor.new_zeros(*new_shape, **kwargs)

    return torch.compile(custom_op_func, backend=fx_backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "size_format,new_shape",
    [
        ("tuple", (5, 5)),
        ("scalar", (5, 5)),
        ("tuple", (10, 20, 25)),
        ("scalar", (10, 20, 25)),
        ("tuple", ()),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_new_zeros_basic(new_shape, dtype, size_format):
    """
    Feature: Test torch.Tensor.new_zeros
    Description: Test new_zeros with tuple size and unpacked scalar sizes
    Expectation: Output shape, dtype, device, and values are correct
    """
    self_tensor = torch.randn(10, 10, dtype=dtype).npu()
    eager_out = _new_zeros_eager(self_tensor, new_shape, size_format, dtype)
    compiled_out = _get_new_zeros_compiled(size_format)(self_tensor, new_shape, dtype)

    AssertRtolEqual(eager_out.cpu(), compiled_out.cpu())
    assert compiled_out.shape == torch.Size(new_shape)
    assert compiled_out.dtype == dtype
    assert compiled_out.device.type == "npu"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("size_format", ["tuple", "scalar"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_new_zeros_int_dtype(dtype, size_format):
    """
    Feature: Test torch.Tensor.new_zeros with integer dtypes
    Description: Test new_zeros with int32 and int64 dtypes
    Expectation: Output dtype and values are correct
    """
    new_shape = (32, 64)
    self_tensor = torch.randn(64, 128, dtype=torch.float32).npu()
    eager_out = _new_zeros_eager(self_tensor, new_shape, size_format, dtype)
    compiled_out = _get_new_zeros_compiled(size_format)(self_tensor, new_shape, dtype)

    AssertRtolEqual(eager_out.cpu(), compiled_out.cpu())
    assert compiled_out.dtype == dtype
    assert compiled_out.device.type == "npu"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "size_format,new_shape",
    [
        ("tuple", ()),
        ("tuple", (5, 6, 7)),
        ("scalar", (5, 6, 7)),
    ],
)
def test_new_zeros_same_dtype(new_shape, size_format):
    """
    Feature: Test torch.Tensor.new_zeros with same dtype as self
    Description: Test new_zeros inheriting dtype from self tensor
    Expectation: Output keeps self dtype
    """
    dtype = torch.float32
    self_tensor = torch.randn(10, dtype=dtype).npu()
    eager_out = _new_zeros_eager(self_tensor, new_shape, size_format)
    compiled_out = _get_new_zeros_compiled(size_format)(self_tensor, new_shape)

    AssertRtolEqual(eager_out.cpu(), compiled_out.cpu())
    assert compiled_out.dtype == dtype
    assert compiled_out.shape == torch.Size(new_shape)
    assert compiled_out.device.type == "npu"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("size_format", ["tuple", "scalar"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_new_zeros_different_self_dtype(dtype, size_format):
    """
    Feature: Test torch.Tensor.new_zeros with different self tensor dtype
    Description: Test new_zeros when self tensor has different dtype than output
    Expectation: Output dtype matches the specified dtype, not self's dtype
    """
    new_shape = (20, 20)
    self_dtype = torch.bfloat16 if dtype == torch.float16 else torch.float16
    self_tensor = torch.randn(10, 10, dtype=self_dtype).npu()
    eager_out = _new_zeros_eager(self_tensor, new_shape, size_format, dtype)
    compiled_out = _get_new_zeros_compiled(size_format)(self_tensor, new_shape, dtype)

    AssertRtolEqual(eager_out.cpu(), compiled_out.cpu())
    assert compiled_out.dtype == dtype
    assert compiled_out.dtype != self_tensor.dtype
    assert compiled_out.device.type == "npu"
