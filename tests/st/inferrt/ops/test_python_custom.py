import pytest
import torch
from typing import List

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual
from mrt.torch.fx_backend import backend as fx_backend
from mrt.torch.fx_mlir_backend import backend as mlir_backend


try:
    from vllm.utils import direct_register_custom_op, supports_custom_op
    VLLM_INSTALLED = True
except ImportError:
    VLLM_INSTALLED = False
    direct_register_custom_op = None
    supports_custom_op = None


def scale_and_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x * 1.5 + bias


def scale_and_bias_fake(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


def mixed_types_op(
    x: torch.Tensor,
    scale: float,
    shift: int,
    flag: bool,
    label: str,
    in_shapes: List[int],
) -> torch.Tensor:
    y = x * scale + shift
    if flag:
        y = y * scale + shift
    else:
        y = y / scale - shift
    assert isinstance(label, str)
    for in_shape in in_shapes:
        y = y + in_shape
    return y


def mixed_types_op_fake(
    x: torch.Tensor,
    scale: float,
    shift: int,
    flag: bool,
    label: str,
    in_shapes: List[int],
) -> torch.Tensor:
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


if VLLM_INSTALLED:
    direct_register_custom_op(
        op_name="scale_and_bias",
        op_func=scale_and_bias,
        fake_impl=scale_and_bias_fake,
        mutates_args=[],
        dispatch_key="PrivateUse1",
    )
    direct_register_custom_op(
        op_name="mixed_types_op",
        op_func=mixed_types_op,
        fake_impl=mixed_types_op_fake,
        mutates_args=[],
        dispatch_key="PrivateUse1",
    )


def _should_skip_test():
    """Check if test should be skipped."""
    if not VLLM_INSTALLED:
        return True
    return not supports_custom_op()


def get_op_func_compiled(backend):
    def custom_op_func(x, bias):
        return torch.ops.vllm.scale_and_bias(x, bias)
    return torch.compile(custom_op_func, backend=backend)


@pytest.mark.skipif(
    _should_skip_test(),
    reason="requires vllm installed and torch versions >=2.4.0 with custom op API"
)
@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [[128, 4096], [32, 1024]])
@pytest.mark.parametrize("backend", [fx_backend, mlir_backend])
def test_python_custom_op(shape, backend):
    """
    Feature: Test python custom op
    Description: Test python custom op with torch.compile
    Expectation: The result is correct
    """
    x_cpu = torch.randn(shape, dtype=torch.float16)
    bias_cpu = torch.randn(shape, dtype=torch.float16)

    x_npu = x_cpu.clone().npu()
    bias_npu = bias_cpu.clone().npu()

    cpu_output = scale_and_bias(x_cpu, bias_cpu)
    op_func_compiled = get_op_func_compiled(backend)
    npu_output = op_func_compiled(x_npu, bias_npu)

    AssertRtolEqual(cpu_output, npu_output.detach().cpu())


def _get_mixed_types_compiled(backend):
    def custom_op_func(x, scale, shift, flag, label, in_shapes):
        return torch.ops.vllm.mixed_types_op(x, scale, shift, flag, label, in_shapes)
    return torch.compile(custom_op_func, backend=backend)


@pytest.mark.skipif(
    _should_skip_test(),
    reason="requires vllm installed and torch versions >=2.4.0 with custom op API"
)
@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "shape,scale,shift,flag,label,in_shapes",
    [
        ([128, 4096], 1.25, 2, True, "python_custom_op", [30, 20, 10]),
        ([32, 1024], 1.25, 1, False, "mixed_types", [16, 8]),
    ],
)
@pytest.mark.parametrize("backend", [fx_backend, mlir_backend])
def test_python_custom_op_mixed_types(shape, scale, shift, flag, label, in_shapes, backend):
    """
    Feature: Test python custom op mixed types conversion
    Description: Verify ValueToPyData supports tensor, float, int, bool, string, tuple
    Expectation: The result is correct and type/value checks in op pass
    """
    x_cpu = torch.randn(shape, dtype=torch.float16)
    x_npu = x_cpu.clone().npu()

    cpu_output = mixed_types_op(x_cpu, scale, shift, flag, label, in_shapes)
    op_func_compiled = _get_mixed_types_compiled(backend)
    npu_output = op_func_compiled(x_npu, scale, shift, flag, label, in_shapes)

    AssertRtolEqual(cpu_output, npu_output.detach().cpu())
