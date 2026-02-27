"""Tests for aclnn swiglu operation."""

import pytest
import torch

from mrt.torch.fx_mlir_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def missing_mfusion():
    try:
        import mfusion  # pylint: disable=import-outside-toplevel,unused-import
    except ModuleNotFoundError:
        return True
    return False


class TwoMulModule(torch.nn.Module):
    """Module with two mul ops that can be fused into a DVM kernel."""

    def forward(self, x, y):
        z = x * y
        out = z * x
        return out


@pytest.mark.skipif(missing_mfusion(), reason="mfusion not installed")
@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_mfusion_two_mul_fusion():
    """
    Feature: mfusion integration with DVM
    Description: Two mul ops should be fused into a single DVM kernel
    Expectation: The result matches eager mode execution
    """

    module = TwoMulModule()
    compiled_module = torch.compile(module, backend=backend)

    x = torch.randn(4, 8, device="npu", dtype=torch.float16)
    y = torch.randn(4, 8, device="npu", dtype=torch.float16)

    expected = module(x, y)
    result = compiled_module(x, y)

    AssertRtolEqual(result, expected)
    print("test_mfusion_two_mul_fusion passed.")
