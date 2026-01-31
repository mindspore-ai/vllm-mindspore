"""Tests for MRT MLIR backend with CPU environment.

CI environment restriction: CPU-only testing available
Note: Disabling NPU backends to prevent torch_npu related import errors
"""
# pylint: disable=wrong-import-position, ungrouped-imports
import os
os.environ["USE_NPU"] = "0"
os.environ["USE_ASCEND"] = "0"
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import pytest
import torch
from tests.common import HasTorchNpu
if HasTorchNpu():
    import torch_npu    # pylint: disable=unused-import
import mrt.torch.fx_mlir_backend as backend
from tests.mark_utils import arg_mark


def missing_torch_mlir():
    try:
        import torch_mlir  # pylint: disable=import-outside-toplevel,unused-import
    except ModuleNotFoundError:
        return True
    return False


@pytest.mark.skip(reason="unsupported matmul op")
@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_reshape():
    """
    Feature: MRT backend
    Description: Test torch.compile with MRT backend
    Expectation: Compiled function produces same results as original function in both pipeline modes
    """

    def foo(x, y):
        a = torch.reshape(y, (x.shape[1], -1))
        return torch.matmul(x, a)

    opt_foo = torch.compile(foo, backend=backend)

    x = torch.randn(2, 2)
    y = torch.arange(4.0)
    bar = foo(x, y)
    opt_bar = opt_foo(x, torch.arange(6.0))
    opt_bar = opt_foo(x, torch.arange(8.0))
    opt_bar = opt_foo(x, y)

    assert torch.equal(opt_bar, bar), f"\nopt_bar={opt_bar}\nbar={bar}"
    print("The result is correct. 'mrt' backend has been installed successfully.")


@pytest.mark.skipif(missing_torch_mlir(), reason="not install torch_mlir")
@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_mul():
    """
    Feature: MRT backend
    Description: Test torch.compile with MRT backend
    Expectation: Compiled function produces same results as original function in both pipeline modes
    """

    def foo(x, y):
        return x * y

    opt_foo = torch.compile(foo, backend=backend)

    x = torch.randn(2, 2)
    y = torch.randn(2, 2)
    bar = foo(x, y)
    opt_bar = opt_foo(torch.randn(3, 3), torch.randn(3, 3))
    opt_bar = opt_foo(torch.randn(5, 4), torch.randn(5, 4))
    opt_bar = opt_foo(x, y)

    assert torch.equal(opt_bar, bar), f"\nopt_bar={opt_bar}\nbar={bar}"
    print("The result is correct. 'mrt' backend has been installed successfully.")
