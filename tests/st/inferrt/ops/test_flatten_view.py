"""Tests for aclnn flatten operation."""
import re

import pytest
import torch
import torch._dynamo.config as dynamo_config

from ms_inferrt.torch import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual

# Avoid Dynamo recompile limit exhaustion when fullgraph=True: batch test runs
# create many compiled functions sharing the same code object, each with different
# input shapes, which can exceed the default cache_size_limit (8) and cause
# FailOnRecompileLimitHit. Raise the limit to ensure all cases go through InferRT
# rather than falling back to eager mode.
dynamo_config.cache_size_limit = 64

_FLATTEN_VIEW_ERR = r"Flatten (view|output) shape .* is not compatible"


def op_func(input_self_tensor, start_dim = 0, end_idx = -1):
    return input_self_tensor.flatten(start_dim, end_idx)

def get_op_func_compiled():
    def custom_op_func(input_self_tensor, start_dim = 0, end_idx = -1):
        return input_self_tensor.flatten(start_dim, end_idx)
    return torch.compile(custom_op_func, backend=backend, fullgraph=True)


def build_non_contiguous_tensor(input_self_tensor, pattern):
    """Build a deterministic non-contiguous view tensor from a base input tensor."""
    if pattern == "permute_0132":
        return input_self_tensor.permute(0, 1, 3, 2)
    if pattern == "transpose_23":
        return input_self_tensor.transpose(2, 3)
    if pattern == "last_dim_narrow":
        return input_self_tensor[..., 1:]
    if pattern == "dim1_narrow":
        return input_self_tensor[:, 1:, ...]
    if pattern == "dim2_narrow":
        return input_self_tensor[:, :, 1:, ...]
    if pattern == "dim1_narrow_transpose_01":
        return input_self_tensor[:, 1:, ...].transpose(0, 1)
    if pattern == "transpose_01":
        return input_self_tensor.transpose(0, 1)
    raise ValueError(f"unsupported pattern: {pattern}")


def assert_nonview_flatten_behavior(op_func_compiled, input_self_tensor_npu, cpu_output):
    """
    InferRT may either:
    1) reject non-view flatten geometry (no contiguous fallback), or
    2) run successfully by materializing contiguous layout internally.
    """
    try:
        npu_output = op_func_compiled(input_self_tensor_npu)
    except RuntimeError as runtime_error:
        assert re.search(_FLATTEN_VIEW_ERR, str(runtime_error))
        return
    AssertRtolEqual(cpu_output, npu_output.detach().cpu())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [[4, 5, 6, 7, 8], [7, 6, 8, 10]])
@pytest.mark.parametrize("start_idx", [0, 1])
@pytest.mark.parametrize("end_idx", [-1, 3])
def test_flatten(shape, start_idx, end_idx):
    """
    Feature: Test aclnn flatten
    Description: Test aclnn flatten with bf16 inputs
    Expectation: The result is correct
    """

    self_tensor = torch.rand(shape, dtype=torch.bfloat16)
    self_tensor_npu = self_tensor.npu()

    cpu_output0  = op_func(self_tensor, start_dim=start_idx, end_idx=end_idx)
    op_func_compiled = get_op_func_compiled()
    npu_output0 = op_func_compiled(self_tensor_npu, start_dim=start_idx, end_idx=end_idx)
    npu_output_opt0 = npu_output0.detach().cpu()
    AssertRtolEqual(cpu_output0, npu_output_opt0)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_flatten_permute_view_input():
    """
    Feature: Test aclnn flatten with non-contiguous (permuted) input
    Description: Flatten on permute(0,1,3,2) tensor where dims [0,1] form a contiguous group
    Expectation: The result is correct
    """
    # permute(0,1,3,2) on [3,4,5,6] -> [3,4,6,5], strides (120,30,1,6)
    # dims 0,1 are contiguous: strides[0]=120 == shape[1]*strides[1]=4*30=120
    self_tensor = torch.rand(3, 4, 5, 6, dtype=torch.bfloat16)
    permuted = self_tensor.permute(0, 1, 3, 2)
    permuted_npu = permuted.npu()

    cpu_output = op_func(permuted, start_dim=0, end_idx=1)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(permuted_npu, start_dim=0, end_idx=1)
    npu_output_cpu = npu_output.detach().cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_flatten_slice_offset_input():
    """
    Feature: Test aclnn flatten with sliced input (non-zero storage offset)
    Description: Flatten on sliced tensor with storage_offset != 0 but contiguous strides
    Expectation: The result is correct
    """
    self_tensor = torch.rand(3, 4, 5, 6, dtype=torch.bfloat16)
    sliced = self_tensor[1:]  # shape [2,4,5,6], contiguous strides, storage_offset != 0
    sliced_npu = sliced.npu()

    cpu_output = op_func(sliced, start_dim=1, end_idx=2)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(sliced_npu, start_dim=1, end_idx=2)
    npu_output_cpu = npu_output.detach().cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_flatten_2d_slice_input():
    """
    Feature: Test aclnn flatten with 2D sliced input
    Description: Flatten on a sliced 2D tensor with non-zero storage offset
    Expectation: The result is correct
    """
    self_tensor = torch.rand(10, 20, dtype=torch.bfloat16)
    sliced = self_tensor[2:8, 3:15]  # shape [6, 12], strides (20, 1), offset=43
    sliced_npu = sliced.npu()

    cpu_output = op_func(sliced, start_dim=0, end_idx=-1)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(sliced_npu, start_dim=0, end_idx=-1)
    npu_output_cpu = npu_output.detach().cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_flatten_permute_partial_flatten():
    """
    Feature: Test aclnn flatten on permuted tensor with partial flatten range
    Description: Flatten on fully permuted non-contiguous input cannot be represented as view
    Expectation: RuntimeError is raised (InferRT flatten_view has no contiguous fallback)
    """
    # permute(0,1,3,2) on [3,4,5,6] -> [3,4,6,5], strides (120,30,1,6)
    # flatten(2,3): strides[2]=1, shape[3]*strides[3]=5*6=30 != 1, NOT view-compatible
    # But PyTorch will make it contiguous first, so the result should still be correct
    self_tensor = torch.rand(3, 4, 5, 6, dtype=torch.bfloat16)
    permuted = self_tensor.permute(0, 1, 3, 2)
    permuted_npu = permuted.npu()

    # flatten(0,-1) on the non-contiguous tensor:
    # eager PyTorch succeeds by contiguous fallback, while InferRT flatten_view should raise.
    # Use fullgraph=True to avoid Dynamo recompile limit exhaustion from earlier tests
    # causing fallback to eager mode (which would not raise).
    op_func_compiled = torch.compile(op_func, backend=backend, fullgraph=True)
    with pytest.raises(RuntimeError, match=_FLATTEN_VIEW_ERR):
        op_func_compiled(permuted_npu, start_dim=0, end_idx=-1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "shape, slice_pattern, start_idx, end_idx",
    [
        ((3, 4, 5, 6), "last_dim_narrow", 0, 1),
        ((4, 5, 6, 7), "dim1_narrow", 2, 3),
        ((2, 3, 4, 5, 6), "dim2_narrow", 3, 4),
        ((4, 5, 6, 7), "transpose_01", -2, -1),
        ((3, 4, 5, 6), "permute_0132", -4, -3),
        ((3, 4, 5, 6), "dim1_narrow_transpose_01", 2, 3),
        ((3, 4, 5, 6), "dim2_narrow", -2, -1),
    ],
)
def test_flatten_noncontiguous_view_compatible(shape, slice_pattern, start_idx, end_idx):
    """
    Feature: flatten view on non-contiguous input
    Description: Non-contiguous tensors can still flatten by view when stride geometry is compatible
    Expectation: Compiled result equals eager result
    """
    self_tensor = torch.rand(shape, dtype=torch.bfloat16)
    self_tensor_npu = self_tensor.npu()
    non_contiguous = build_non_contiguous_tensor(self_tensor, slice_pattern)
    assert not non_contiguous.is_contiguous()

    def custom_op_func(input_self_tensor):
        inter = build_non_contiguous_tensor(input_self_tensor, slice_pattern)
        return inter.flatten(start_idx, end_idx)

    cpu_output = custom_op_func(self_tensor)
    assert cpu_output.data_ptr() == non_contiguous.data_ptr()

    op_func_compiled = torch.compile(custom_op_func, backend=backend, fullgraph=True)
    npu_output = op_func_compiled(self_tensor_npu)
    npu_output_cpu = npu_output.detach().cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "shape, transform_pattern, start_idx, end_idx",
    [
        ((3, 4, 5, 6), "permute_0132", 2, 3),
        ((4, 5, 6, 7), "transpose_01", 0, 1),
    ],
)
def test_flatten_noncontiguous_view_incompatible(shape, transform_pattern, start_idx, end_idx):
    """
    Feature: flatten non-view path on non-contiguous input
    Description: Non-contiguous intermediate whose target flatten shape cannot infer legal strides
    Expectation: InferRT either reports unsupported layout or matches eager output
    """
    self_tensor = torch.rand(shape, dtype=torch.bfloat16)
    self_tensor_npu = self_tensor.npu()

    transformed = build_non_contiguous_tensor(self_tensor, transform_pattern)
    assert not transformed.is_contiguous()
    cpu_output = transformed.flatten(start_idx, end_idx)
    assert cpu_output.data_ptr() != transformed.data_ptr()

    def custom_op_func(input_self_tensor):
        inter = build_non_contiguous_tensor(input_self_tensor, transform_pattern)
        return inter.flatten(start_idx, end_idx)

    op_func_compiled = torch.compile(custom_op_func, backend=backend, fullgraph=True)
    assert_nonview_flatten_behavior(op_func_compiled, self_tensor_npu, cpu_output)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_flatten_0d_to_1d():
    """
    Feature: 0D flatten
    Description: Flatten a scalar tensor should return a [1] tensor
    Expectation: Shape/value are consistent with eager
    """
    self_tensor = torch.tensor(2.5, dtype=torch.bfloat16)
    self_tensor_npu = self_tensor.npu()

    cpu_output = op_func(self_tensor, start_dim=0, end_idx=-1)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(self_tensor_npu, start_dim=0, end_idx=-1)
    npu_output_cpu = npu_output.detach().cpu()

    assert tuple(cpu_output.shape) == (1,)
    assert tuple(npu_output_cpu.shape) == (1,)
    AssertRtolEqual(cpu_output, npu_output_cpu)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_flatten_invalid_dim_order_matches_eager():
    """
    Feature: flatten invalid dim order
    Description: Verify start_dim > end_dim keeps the same failure semantics as eager mode
    Expectation: Eager and compiled execution both raise RuntimeError
    """

    def func(x, start_dim, end_dim):
        return x.flatten(start_dim, end_dim)

    x = torch.rand(2, 3, 4, 5, dtype=torch.bfloat16).npu()
    compiled_func = torch.compile(func, backend=backend, fullgraph=True)

    with pytest.raises(RuntimeError, match=r"start_dim cannot come after end_dim"):
        func(x, 2, 1)
    with pytest.raises(RuntimeError, match=r"start_dim cannot come after end_dim"):
        compiled_func(x, 2, 1)
