"""Tests for linear operation."""
import pytest
import torch
import torch.nn.functional as F
import torch_npu

from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(inputs, weight, bias=None):
    out = F.linear(inputs, weight, bias)
    return out


def get_op_func_compiled():
    def custom_op_func(inputs, weight, bias=None):
        return F.linear(inputs, weight, bias)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("batch_size", [10, 20, 30])
@pytest.mark.parametrize("in_features", [15, 25, 35])
@pytest.mark.parametrize("out_features", [24, 88, 108])
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("has_bias", [False, True])
def test_linear(dtype, batch_size, in_features, out_features, has_bias):
    """
    Feature: Test linear
    Description: Test linear with fp16/bf16 inputs
    Expectation: The result is correct
    """

    x_input = torch.randn(batch_size, in_features).to(dtype)
    weight = torch.randn(out_features, in_features).to(dtype)

    bias = None
    if has_bias:
        bias = torch.randn(out_features).to(dtype)

    x_input_npu = x_input.npu()
    weight_npu = weight.npu()
    if has_bias:
        bias_npu = bias.npu()
    else:
        bias_npu = None

    expected_output0_npu = op_func(x_input_npu, weight_npu, bias_npu)
    op_func_compiled = get_op_func_compiled()

    npu_output0 = op_func_compiled(x_input_npu, weight_npu, bias_npu)
    npu_output = npu_output0.detach().cpu()
    AssertRtolEqual(expected_output0_npu.detach().cpu(), npu_output)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("batch_size", [10, 20, 30])
@pytest.mark.parametrize("in_features", [15, 25, 35])
@pytest.mark.parametrize("out_features", [24, 88, 108])
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("has_bias", [False, True])
def test_linear_3d(dtype, batch_size, in_features, out_features, has_bias):
    """
    Feature: Test linear
    Description: Test linear with fp16/bf16 inputs
    Expectation: The result is correct
    """

    x_input = torch.randn(batch_size, batch_size, in_features).to(dtype)
    weight = torch.randn(out_features, in_features).to(dtype)

    bias = None
    if has_bias:
        bias = torch.randn(out_features).to(dtype)

    x_input_npu = x_input.npu()
    weight_npu = weight.npu()
    if has_bias:
        bias_npu = bias.npu()
    else:
        bias_npu = None

    expected_output0_npu = op_func(x_input_npu, weight_npu, bias_npu)
    op_func_compiled = get_op_func_compiled()

    npu_output0 = op_func_compiled(x_input_npu, weight_npu, bias_npu)
    npu_output = npu_output0.detach().cpu()
    AssertRtolEqual(expected_output0_npu.detach().cpu(), npu_output)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("batch_size", [10, 20, 30])
@pytest.mark.parametrize("in_features", [15, 25, 35])
@pytest.mark.parametrize("out_features", [24, 88, 108])
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("has_bias", [False, True])
def test_linear_4d(dtype, batch_size, in_features, out_features, has_bias):
    """
    Feature: Test linear
    Description: Test linear with fp16/bf16 inputs
    Expectation: The result is correct
    """

    x_input = torch.randn(batch_size, batch_size,
                          in_features, in_features).to(dtype)
    weight = torch.randn(out_features, in_features).to(dtype)

    bias = None
    if has_bias:
        bias = torch.randn(out_features).to(dtype)

    x_input_npu = x_input.npu()
    weight_npu = weight.npu()
    if has_bias:
        bias_npu = bias.npu()
    else:
        bias_npu = None

    expected_output0_npu = op_func(x_input_npu, weight_npu, bias_npu)
    op_func_compiled = get_op_func_compiled()

    npu_output0 = op_func_compiled(x_input_npu, weight_npu, bias_npu)
    npu_output = npu_output0.detach().cpu()
    AssertRtolEqual(expected_output0_npu.detach().cpu(), npu_output)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("batch_size", [10, 20, 30])
@pytest.mark.parametrize("in_features", [15, 25, 35])
@pytest.mark.parametrize("out_features", [24, 88, 108])
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("has_bias", [False, True])
def test_linear_nz(dtype, batch_size, in_features, out_features, has_bias):
    """
    Feature: Test linear with NZ format weight
    Description: Test linear with NZ format weight tensor in inferRT backend
    Expectation: The result is correct
    """

    x_input = torch.randn(batch_size, in_features).to(dtype)
    weight = torch.randn(out_features, in_features).to(dtype)

    bias = None
    if has_bias:
        bias = torch.randn(out_features).to(dtype)

    x_input_npu = x_input.npu()
    weight_nd = weight.npu()
    # Convert weight to NZ format (FORMAT_FRACTAL_NZ = 29)
    weight_nz = torch_npu.npu_format_cast(weight_nd, 29).npu()
    if has_bias:
        bias_npu = bias.npu()
    else:
        bias_npu = None

    # Expected output using original weight (for comparison)
    expected_output0_npu = op_func(x_input_npu, weight.npu(), bias_npu)
    op_func_compiled = get_op_func_compiled()

    # Test with NZ format weight
    npu_output0 = op_func_compiled(x_input_npu, weight_nz, bias_npu)
    npu_output = npu_output0.detach().cpu()
    AssertRtolEqual(expected_output0_npu.detach().cpu(), npu_output)
