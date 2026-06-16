"""Tests for torch.nn.functional.layer_norm operation."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ms_inferrt.torch.fx_mlir_backend import backend as mlir_backend
from ms_inferrt.torch.fx_backend import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(x, normalized_shape, weight, bias, eps=1e-5):
    """Reference implementation of layer_norm using PyTorch eager mode."""
    return F.layer_norm(x, normalized_shape, weight, bias, eps)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend, mlir_backend))
def test_layer_norm(backend):
    """
    Feature: Test aclnn layer_norm
    Description: Test aclnn layer_norm with fp32 inputs and different backends
    Expectation: The result is correct
    """

    def custom_op_func(x, normalized_shape, weight, bias, eps=1e-5):
        return F.layer_norm(x, normalized_shape, weight, bias, eps)

    op_func_compiled = torch.compile(custom_op_func, backend=backend)

    normalized_shape = (4,)

    cpu_input = np.random.randn(2, 3, 4).astype(np.float32)
    cpu_weight = np.random.randn(4).astype(np.float32)
    cpu_bias = np.random.randn(4).astype(np.float32)

    cpu_tensor_input = torch.from_numpy(cpu_input)
    cpu_tensor_weight = torch.from_numpy(cpu_weight)
    cpu_tensor_bias = torch.from_numpy(cpu_bias)

    npu_input = cpu_tensor_input.npu()
    npu_weight = cpu_tensor_weight.npu()
    npu_bias = cpu_tensor_bias.npu()

    cpu_output = op_func(cpu_tensor_input, normalized_shape, cpu_tensor_weight, cpu_tensor_bias)

    npu_output = op_func_compiled(npu_input, normalized_shape, npu_weight, npu_bias)

    cpu_output_np = cpu_output.detach().numpy()
    npu_output_np = npu_output.detach().cpu().numpy()

    AssertRtolEqual(cpu_output_np, npu_output_np)

def _make_layer_norm_fx_case(case):
    """Create a layer_norm callable and inputs for the requested FX backend case."""
    input_tensor = torch.randn((2, 4), dtype=torch.float16).npu()
    weight = torch.randn((4,), dtype=torch.float16).npu()
    bias = torch.randn((4,), dtype=torch.float16).npu()

    if case == "functional_hardcoded_kwargs_no_affine":
        def functional_hardcoded_kwargs_no_affine(x):
            return F.layer_norm(x, normalized_shape=(4,), eps=1e-5)

        return functional_hardcoded_kwargs_no_affine, (input_tensor,)

    if case == "functional_kwargs_swapped":
        def functional_kwargs_swapped(x):
            return F.layer_norm(x, eps=1e-5, normalized_shape=(4,))

        return functional_kwargs_swapped, (input_tensor,)

    if case == "functional_all_positional":
        def functional_all_positional(x, w, b):
            return F.layer_norm(x, (4,), w, b, 1e-5)

        return functional_all_positional, (input_tensor, weight, bias)

    if case == "functional_optional_kwargs_swapped":
        def functional_optional_kwargs_swapped(x, w, b):
            return F.layer_norm(x, (4,), bias=b, weight=w, eps=1e-5)

        return functional_optional_kwargs_swapped, (input_tensor, weight, bias)

    if case == "torch_layer_norm_no_affine":
        def torch_layer_norm_no_affine(x):
            return torch.layer_norm(x, (4,), None, None, 1e-5)

        return torch_layer_norm_no_affine, (input_tensor,)

    if case == "torch_layer_norm_all_positional_with_cudnn":
        def torch_layer_norm_all_positional_with_cudnn(x, w, b):
            return torch.layer_norm(x, (4,), w, b, 1e-5, False)

        return torch_layer_norm_all_positional_with_cudnn, (input_tensor, weight, bias)

    if case == "torch_layer_norm_kwargs_swapped_with_cudnn":
        def torch_layer_norm_kwargs_swapped_with_cudnn(x, w, b):
            return torch.layer_norm(x, eps=1e-5, bias=b, weight=w, normalized_shape=(4,), cudnn_enable=True)

        return torch_layer_norm_kwargs_swapped_with_cudnn, (input_tensor, weight, bias)

    raise ValueError(f"Unsupported layer_norm case: {case}")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "case",
    [
        "functional_hardcoded_kwargs_no_affine",
        "functional_kwargs_swapped",
        "functional_all_positional",
        "functional_optional_kwargs_swapped",
        "torch_layer_norm_no_affine",
        "torch_layer_norm_all_positional_with_cudnn",
        "torch_layer_norm_kwargs_swapped_with_cudnn",
    ],
)
def test_layer_norm_fx_backend_frontend_and_arg_variants(case):
    """
    Feature: Test FX backend layer_norm frontend and argument variants
    Description: Verify functional/torch frontends, optional args, kwargs order, and cudnn_enable handling
    Expectation: Compiled result is numerically consistent with eager result
    """
    custom_op_func, inputs = _make_layer_norm_fx_case(case)
    eager_output = custom_op_func(*inputs)

    op_func_compiled = torch.compile(custom_op_func, backend=fx_backend, fullgraph=True)
    compiled_output = op_func_compiled(*inputs)

    torch.testing.assert_close(compiled_output, eager_output, rtol=1e-3, atol=1e-3)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_aten_native_layer_norm_default():
    """
    Feature: Test aten.native_layer_norm.default via fx_backend
    Description: Directly call torch.ops.aten.native_layer_norm.default
    Expectation: Result matches reference (output, mean, rstd)
    """
    input_tensor = torch.randn((1, 70, 2048), dtype=torch.float32).npu()
    weight = torch.randn((2048,), dtype=torch.float32).npu()
    bias = torch.randn((2048,), dtype=torch.float32).npu()

    def native_ln(x, w, b):
        return torch.ops.aten.native_layer_norm.default(x, [2048], w, b, 1e-5)

    eager_out = native_ln(input_tensor, weight, bias)
    op_func_compiled = torch.compile(native_ln, backend=fx_backend, fullgraph=True)
    compiled_out = op_func_compiled(input_tensor, weight, bias)

    for e, c in zip(eager_out, compiled_out):
        torch.testing.assert_close(c, e, rtol=1e-3, atol=1e-3)
