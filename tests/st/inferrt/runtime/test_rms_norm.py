"""Runtime tests for torch.rms_norm lowering in ms_inferrt backend."""

import pytest
import torch
from torch import nn

from ms_inferrt.torch.fx_backend import backend as fx_backend
from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


class ReproModule(nn.Module):
    """Minimal module that reproduces the rms_norm runtime path."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(7168, dtype=torch.bfloat16))
        self.eps = 1e-06

    def forward(self, x, dim_val):
        """Run torch.rms_norm and cast output back to bf16."""
        x_f32 = x.to(torch.float32)
        weight_f32 = self.weight.to(torch.float32)
        out_f32 = torch.rms_norm(x_f32, (dim_val,), weight_f32, self.eps)
        return out_f32.to(torch.bfloat16)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend,))
def test_rms_norm_runtime_lowering(backend):
    """
    Feature: Runtime lowering for torch.rms_norm
    Description: Compile module with ms_inferrt backend and compare with eager result
    Expectation: The compiled result matches eager output
    """
    s0 = 2
    x = torch.randn(s0, 7168, dtype=torch.bfloat16).npu()
    dim_val = 7168
    model = ReproModule().npu()

    compiled_model = torch.compile(model, backend=backend, dynamic=True)
    out = compiled_model(x, dim_val)
    expected = model(x, dim_val)

    AssertRtolEqual(out, expected)
