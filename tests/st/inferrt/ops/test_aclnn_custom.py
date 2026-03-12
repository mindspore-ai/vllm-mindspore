"""Tests for custom aclnn operations."""
import os

import pytest
import torch

import ms_inferrt
from ms_inferrt.torch.fx_backend import backend as fx_backend
from ms_inferrt.torch.fx_mlir_backend import backend as mlir_backend

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
    ms_inferrt.ops.load(name="aclnn_custom_div", sources=[aclnn_div_path], backend="Ascend")

    @torch.library.custom_op("ms_inferrt::custom_div", mutates_args=())
    def custom_div_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is a placeholder for the custom_div operator.")

    # pylint: disable=unused-argument
    @torch.library.register_fake("ms_inferrt::custom_div")
    def _(x, y):
        return x

    def ms_inferrt_custom_div(x, y):
        return torch.ops.ms_inferrt.custom_div(x, y)

    ms_inferrt_custom_div_compiled = torch.compile(ms_inferrt_custom_div, backend=backend)

    x = torch.randn(2, 2).npu()
    y = torch.randn(2, 2).npu()
    result = ms_inferrt_custom_div_compiled(x, y)
    expected = torch.div(x, y)

    assert torch.equal(result, expected), f"\nresult={result}\nexpected={expected}"
    print("The result is correct. Launch aclnn custom op [div] successfully.")


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_custom_ops_api_with_ascend_backend():
    """
    Feature: Test ms_inferrt.ops.compile and ms_inferrt.ops.load_library APIs
    Description: Test ms_inferrt.ops.compile and ms_inferrt.ops.load_library APIs
    Expectation: The result is correct
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cpu_sub_path = os.path.join(script_dir, "aclnn_custom_div.cc")
    lib_path = ms_inferrt.ops.compile(name="custom_div", sources=[cpu_sub_path], backend="Ascend")
    result = ms_inferrt.ops.load_library(lib_path)
    assert result is True, "ms_inferrt.ops.load_library failed"
