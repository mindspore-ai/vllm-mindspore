import os
"""
CI environment restriction: CPU-only testing available
Note: Disabling NPU backends to prevent torch_npu related import errors
"""
os.environ["USE_NPU"] = "0"
os.environ["USE_ASCEND"] = "0"
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch
import pytest
from mrt.torch import backend
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_backend(pipeline, monkeypatch):
    """
    Feature: MRT backend
    Description: Test torch.compile with MRT backend when MRT_ENABLE_PIPELINE is enabled/disabled
    Expectation: Compiled function produces same results as original function in both pipeline modes
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")

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
