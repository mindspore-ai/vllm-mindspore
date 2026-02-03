"""Tests for CPU custom operations."""
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
import mrt
from mrt.torch.fx_backend import backend as fx_backend
from mrt.torch.fx_mlir_backend import backend as mlir_backend
from tests.mark_utils import arg_mark


def missing_torch_mlir():
    try:
        import torch_mlir  # pylint: disable=import-outside-toplevel,unused-import
    except ModuleNotFoundError:
        return True
    return False


@pytest.mark.skipif(missing_torch_mlir(), reason="not install torch_mlir")
@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
           card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend, mlir_backend))
def test_cpu_custom_add_op(backend):
    """
    Feature: Check CPU custom add op launch
    Description: Check CPU custom add op launch with cache
    Expectation: The result is correct
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cpu_add_path = os.path.join(script_dir, "cpu_custom_add.cc")
    mrt.ops.load(name="custom_add", sources=[cpu_add_path], backend="CPU")

    @torch.library.custom_op("mrt::custom_add", mutates_args=())
    def custom_add_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _ = x  # pylint: disable=unused-argument
        _ = y  # pylint: disable=unused-argument
        raise NotImplementedError("This is a placeholder for the custom_add operator.")

    @torch.library.register_fake("mrt::custom_add")
    def _custom_add_fake(x, y):
        _ = y  # pylint: disable=unused-argument
        return x

    def mrt_custom_add(x, y):
        return torch.ops.mrt.custom_add(x, y)

    mrt_custom_add_compiled = torch.compile(mrt_custom_add, backend=backend)

    x = torch.randn(2, 2).cpu()
    y = torch.randn(2, 2).cpu()
    result = mrt_custom_add_compiled(x, y)
    expected = torch.add(x, y)

    assert torch.equal(result, expected), f"\nresult={result}\nexpected={expected}"
    print("The result is correct. Launch CPU custom op [add] successfully.")


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_custom_ops_api_with_cpu_backend():
    """
    Feature: Test mrt.ops.compile and mrt.ops.load_library APIs
    Description: Test mrt.ops.compile and mrt.ops.load_library APIs
    Expectation: The result is correct
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cpu_sub_path = os.path.join(script_dir, "cpu_custom_add.cc")
    lib_path = mrt.ops.compile(name="custom_add", sources=[cpu_sub_path], backend="CPU")
    result = mrt.ops.load_library(lib_path)
    assert result is True, "mrt.ops.load_library failed"
