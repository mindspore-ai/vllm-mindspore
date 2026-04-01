"""Tests for aclnn dequant_swiglu_quant operation."""

import pytest
import torch
import torch_npu

from ms_inferrt.torch.fx_backend import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
@pytest.mark.parametrize("backend", (fx_backend, ))
def test_npu_dequant_swiglu_quant(backend):
    """
    Feature: Check npu_dequant_swiglu_quant op launch
    Description: Verify dequant_swiglu_quant with V1-compatible arguments
    Expectation: The result matches torch_npu native implementation
    """

    def dequant_swiglu_quant_func(x, weight_scale, activation_scale, quant_scale, group_index):
        return torch_npu.npu_dequant_swiglu_quant(
            x,
            weight_scale=weight_scale,
            activation_scale=activation_scale,
            bias=None,
            quant_scale=quant_scale,
            quant_offset=None,
            group_index=group_index,
            activate_left=True,
            quant_mode=1,
            swiglu_mode=0,
            clamp_limit=7.0,
            glu_alpha=1.702,
            glu_bias=1.0,
        )

    compiled_func = torch.compile(dequant_swiglu_quant_func, backend=backend)

    x = torch.randint(-10, 10, [128, 256], dtype=torch.int32).npu()
    weight_scale = torch.randn([256], dtype=torch.float32).npu()
    activation_scale = torch.randn([128, 1], dtype=torch.float32).npu()
    quant_scale = torch.randn([1, 128], dtype=torch.float32).npu()
    group_index = torch.tensor([128], dtype=torch.int64).npu()

    y, scale = compiled_func(x, weight_scale, activation_scale, quant_scale, group_index)
    expected_y, expected_scale = dequant_swiglu_quant_func(x, weight_scale, activation_scale, quant_scale, group_index)

    AssertRtolEqual(y, expected_y, prec=1e-5, prec16=1e-5)
    AssertRtolEqual(scale, expected_scale, prec=1e-4, prec16=1e-4)
