"""Tests for custom aclnn operations."""
import os

import pytest
import torch

import mrt
from mrt.torch.fx_backend import backend as fx_backend
from mrt.torch.fx_mlir_backend import backend as mlir_backend

from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend, mlir_backend))
def test_aclnn_custom_div_op(backend):
    """
    Feature: Check aclnn custom div op launch
    Description: Check aclnn custom div op launch with cache
    Expectation: The result is correct
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    aclnn_div_path = os.path.join(script_dir, "aclnn_custom_div.cc")
    mrt.ops.load(name="aclnn_custom_div", sources=[aclnn_div_path], backend="Ascend")

    @torch.library.custom_op("mrt::custom_div", mutates_args=())
    def custom_div_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is a placeholder for the custom_div operator.")

    # pylint: disable=unused-argument
    @torch.library.register_fake("mrt::custom_div")
    def _(x, y):
        return x

    def mrt_custom_div(x, y):
        return torch.ops.mrt.custom_div(x, y)

    mrt_custom_div_compiled = torch.compile(mrt_custom_div, backend=backend)

    x = torch.randn(2, 2).npu()
    y = torch.randn(2, 2).npu()
    result = mrt_custom_div_compiled(x, y)
    expected = torch.div(x, y)

    assert torch.equal(result, expected), f"\nresult={result}\nexpected={expected}"
    print("The result is correct. Launch aclnn custom op [div] successfully.")


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_custom_ops_api_with_ascend_backend():
    """
    Feature: Test mrt.ops.compile and mrt.ops.load_library APIs
    Description: Test mrt.ops.compile and mrt.ops.load_library APIs
    Expectation: The result is correct
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cpu_sub_path = os.path.join(script_dir, "aclnn_custom_div.cc")
    lib_path = mrt.ops.compile(name="custom_div", sources=[cpu_sub_path], backend="Ascend")
    result = mrt.ops.load_library(lib_path)
    assert result is True, "mrt.ops.load_library failed"
