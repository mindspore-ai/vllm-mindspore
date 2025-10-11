import torch
from mrt.torch import backend

import pytest
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_check_aclnn_op(pipeline, monkeypatch):
    """
    Feature: Check aclnn op launch
    Description: Check aclnn op launch with cache
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    
    def foo(x, y):
        return torch.mul(x, y)

    opt_foo = torch.compile(foo, backend=backend)

    x = torch.randn(2, 2).npu()
    y = torch.randn(2, 2).npu()
    bar = foo(x, y)
    opt_bar = opt_foo(x, y)
    opt_bar1 = opt_foo(x, y)

    assert torch.equal(opt_bar, opt_bar1), f"\nopt_bar={opt_bar}\nopt_bar1={opt_bar1}"
    assert torch.equal(opt_bar, bar), f"\nopt_bar={opt_bar}\nbar={bar}"

    z = torch.randn(3, 3).npu()
    q = torch.randn(3, 3).npu()
    bar2 = foo(z, q)
    opt_bar2 = opt_foo(z, q)
    opt_bar21 = opt_foo(z, q)

    for _ in range(10):
        opt_bar2 = opt_foo(z, q)

    assert torch.equal(opt_bar2, opt_bar21), f"\nopt_bar2={opt_bar2}\nopt_bar21={opt_bar21}"
    assert torch.equal(opt_bar2, bar2), f"\nopt_bar2={opt_bar2}\nbar2={bar2}"

    print("The result is correct. Launch aclnn [mul] successfully.")
