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
Tests for ref (output-input storage alias) support in custom_call operators.

A custom operator declares the alias by overriding Operator::GetOutputInputRefPairs(), which
OpCustomCall forwards to the runtime. The observable effect of a working alias is semantic rather
than pointer identity: the graph runs on its own device buffers, so comparing torch data_ptr()
across the graph boundary is not meaningful (a built-in ref op such as tensor.copy_ behaves the
same way). What a ref operator must guarantee is that it writes into / reads from the storage of
its input instead of a freshly allocated one, which these tests check through observable behaviour.
"""
import os

import torch

import ms_inferrt
from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark


def _load_custom_ref_ops():
    """Compile and load the custom ref/view operators used by these tests."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source = os.path.join(script_dir, "aclnn_custom_ref.cc")
    ms_inferrt.ops.load(name="aclnn_custom_ref", sources=[source], backend="Ascend")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_custom_call_inplace_ref():
    """
    Feature: custom_call ref support (in-place type)
    Description: A custom operator derived from AclnnCustomOperator declares output 0 as an alias of
                 input 0, so the runtime reuses the input storage instead of allocating a new one and
                 the kernel writes through that alias.
    Expectation: The caller's tensor is updated in place, so repeated calls accumulate, and the
                 returned tensor carries the same values as the mutated input.
    """
    _load_custom_ref_ops()

    @torch.library.custom_op("ms_inferrt::custom_inplace_add", mutates_args=("x",))
    def custom_inplace_add(x: torch.Tensor, y: torch.Tensor, alpha: int) -> torch.Tensor:
        raise NotImplementedError("Placeholder for the custom_inplace_add operator.")

    # pylint: disable=unused-argument
    @torch.library.register_fake("ms_inferrt::custom_inplace_add")
    def _(x, y, alpha):
        return torch.empty_like(x)

    def inplace_add(x, y):
        return torch.ops.ms_inferrt.custom_inplace_add(x, y, 1)

    compiled = torch.compile(inplace_add, backend=backend)

    x = torch.randn(4, 8).npu()
    y = torch.ones(4, 8).npu()
    origin = x.cpu().clone()

    result = compiled(x, y)
    assert torch.allclose(x.cpu(), origin + 1), (
        "in-place custom op must write through the alias into the caller's tensor"
    )
    assert torch.allclose(result.cpu(), x.cpu()), "returned tensor must carry the in-place result"

    # Running again must accumulate: this is only possible when the operator really writes into the
    # storage of its input instead of producing a freshly allocated output.
    result = compiled(x, y)
    assert torch.allclose(x.cpu(), origin + 2), "repeated in-place calls must accumulate"
    assert torch.allclose(result.cpu(), x.cpu()), "returned tensor must carry the in-place result"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_custom_call_view_ref():
    """
    Feature: custom_call ref support (view type)
    Description: A custom operator derived from AclnnCustomViewOperator produces a zero-copy view of its
                 input: the output shares the input storage and only gets a new shape / strides /
                 storageOffset, computed by the operator itself in CalcWorkspace().
    Expectation: The output has the narrowed shape and exactly the values of the corresponding slice,
                 which requires the storage offset to be applied correctly.
    """
    _load_custom_ref_ops()

    @torch.library.custom_op("ms_inferrt::custom_narrow_view", mutates_args=())
    def custom_narrow_view(x: torch.Tensor, start: int, length: int) -> torch.Tensor:
        raise NotImplementedError("Placeholder for the custom_narrow_view operator.")

    # pylint: disable=unused-argument
    @torch.library.register_fake("ms_inferrt::custom_narrow_view")
    def _(x, start, length):
        return x.new_empty((length,) + tuple(x.shape[1:]))

    start, length = 2, 3

    def narrow_view(x):
        return torch.ops.ms_inferrt.custom_narrow_view(x, start, length)

    compiled = torch.compile(narrow_view, backend=backend)

    x = torch.randn(8, 4).npu()
    expected = x.cpu()[start:start + length]

    result = compiled(x)

    assert tuple(result.shape) == (length, 4), f"unexpected view shape: {tuple(result.shape)}"
    assert torch.allclose(result.cpu(), expected), f"\nresult={result.cpu()}\nexpected={expected}"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_custom_call_view_ref_offset_zero():
    """
    Feature: custom_call ref support (view type, degenerate offset)
    Description: A view starting at 0 spanning the whole tensor must still round-trip correctly.
    Expectation: The output equals the input.
    """
    _load_custom_ref_ops()

    if not hasattr(torch.ops.ms_inferrt, "custom_narrow_view"):
        @torch.library.custom_op("ms_inferrt::custom_narrow_view", mutates_args=())
        def custom_narrow_view(x: torch.Tensor, start: int, length: int) -> torch.Tensor:
            raise NotImplementedError("Placeholder for the custom_narrow_view operator.")

        # pylint: disable=unused-argument
        @torch.library.register_fake("ms_inferrt::custom_narrow_view")
        def _(x, start, length):
            return x.new_empty((length,) + tuple(x.shape[1:]))

    def full_view(x):
        return torch.ops.ms_inferrt.custom_narrow_view(x, 0, 6)

    compiled = torch.compile(full_view, backend=backend)

    x = torch.randn(6, 4).npu()
    result = compiled(x)

    assert tuple(result.shape) == (6, 4), f"unexpected view shape: {tuple(result.shape)}"
    assert torch.allclose(result.cpu(), x.cpu()), "full-range view must equal the input"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_custom_call_non_ref_regression():
    """
    Feature: custom_call ref support (regression)
    Description: A custom operator that declares no ref pair must keep the previous behaviour, i.e. it
                 allocates its own output and leaves the inputs untouched.
    Expectation: The result is correct and the inputs are unchanged.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    div_source = os.path.join(script_dir, "aclnn_custom_div.cc")
    ms_inferrt.ops.load(name="aclnn_custom_div", sources=[div_source], backend="Ascend")

    @torch.library.custom_op("ms_inferrt::custom_div", mutates_args=())
    def custom_div_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Placeholder for the custom_div operator.")

    # pylint: disable=unused-argument
    @torch.library.register_fake("ms_inferrt::custom_div")
    def _(x, y):
        return x

    def div(x, y):
        return torch.ops.ms_inferrt.custom_div(x, y)

    compiled = torch.compile(div, backend=backend)

    x = torch.randn(2, 2).npu()
    y = torch.randn(2, 2).npu()
    x_origin = x.cpu().clone()
    expected = torch.div(x, y)

    result = compiled(x, y)

    assert torch.equal(result, expected), f"\nresult={result}\nexpected={expected}"
    assert torch.allclose(x.cpu(), x_origin), "a non-ref custom op must not modify its input"
