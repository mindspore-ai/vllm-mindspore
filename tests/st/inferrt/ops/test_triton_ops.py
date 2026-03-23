"""Test Triton operations."""
import pytest
import torch
import triton
import triton.language as tl
from torch._library import capture_triton
from ms_inferrt.torch.fx_backend import backend as fx_backend
from ms_inferrt.torch.fx_mlir_backend import backend as mlir_backend
from tests.mark_utils import arg_mark

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    n_elements: tl.constexpr,
    block_size: tl.constexpr,
):
    """Triton kernel for adding two tensors element-wise."""
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y
    tl.store(z_ptr + offsets, z, mask=mask)

@torch.library.triton_op("triton_ops::add", mutates_args=())
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Add two tensors element-wise using Triton kernel."""
    output = torch.zeros_like(x)
    n_elements = x.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["block_size"]),)

    capture_triton(add_kernel)[grid](x, y, output, n_elements, block_size=128)
    return output

def add_triton_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.mul(x, y)
    out = add(z, x)
    return out

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend, mlir_backend))
def test_triton_add_dynamic_shape(backend):
    """
    Feature: Test triton_ops::add
    Description: Test triton_ops::add with dynamic shape
    Expectation: The result is correct
    """
    x1 = torch.randn([2, 3, 4], dtype=torch.float32).npu()
    y1 = torch.randn([2, 3, 4], dtype=torch.float32).npu()
    compiled_net = torch.compile(add_triton_func, backend=backend)
    z1 = compiled_net(x1, y1)
    expected1 = add_triton_func(x1, y1)
    assert torch.allclose(z1, expected1)

    x2 = torch.randn([32, 64], dtype=torch.float32).npu()
    y2 = torch.randn([32, 64], dtype=torch.float32).npu()
    z2 = compiled_net(x2, y2)
    expected2 = add_triton_func(x2, y2)
    assert torch.allclose(z2, expected2)
