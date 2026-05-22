"""Tests for torch.chunk operation."""
import pytest
import torch
import torch._dynamo

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(input_tensor, chunks, dim=0):
    return input_tensor.chunk(chunks, dim)


def get_op_func_compiled():
    def custom_op_func(input_tensor, chunks, dim=0):
        return input_tensor.chunk(chunks, dim)

    return torch.compile(custom_op_func, backend=backend)


def _assert_chunk_outputs_equal(expected, actual):
    assert len(expected) == len(actual)
    for cpu_out, npu_out in zip(expected, actual):
        AssertRtolEqual(cpu_out, npu_out.detach().cpu())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape, dim, chunks", [
    ([128, 4096], 0, 2),
    ([33, 1024], 0, 4),   # dim size not divisible by chunks
    ([16, 33], 1, 4),     # split along non-leading dim, not divisible
])
def test_chunk(shape, dim, chunks):
    """
    Feature: Test torch.chunk
    Description: Compare compiled NPU chunk results with CPU reference
    Expectation: All chunk outputs match between CPU and NPU
    """
    cpu_input = torch.rand(shape, dtype=torch.bfloat16)
    npu_input = cpu_input.npu()

    cpu_chunks = op_func(cpu_input, chunks, dim=dim)
    op_func_compiled = get_op_func_compiled()
    npu_chunks = op_func_compiled(npu_input, chunks, dim=dim)

    _assert_chunk_outputs_equal(cpu_chunks, npu_chunks)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dim, chunks, shapes", [
    (0, 16, [[5120, 8], [3072, 8]]),     # qwen-style leading dynamic dim
    (0, 4, [[33, 17], [65, 17]]),        # uneven chunk sizes and dynamic dim
    (-1, 4, [[8, 33], [8, 65]]),         # negative dim with uneven chunk sizes
    (0, 64, [[33, 8], [65, 8]]),         # chunks may exceed dim size
])
def test_chunk_dynamic_shape_reuses_graph(dim, chunks, shapes):
    """
    Feature: Test torch.chunk with dynamic chunk sizes
    Description: Keep chunk split sizes symbolic across different input shapes.
    Expectation: Outputs are correct and the compiled graph is reused.
    """
    compile_count = 0

    def counting_backend(gm, example_inputs):
        nonlocal compile_count
        compile_count += 1
        return backend(gm, example_inputs)

    def custom_op_func(input_tensor):
        return input_tensor.chunk(chunks, dim)

    compiled_op = torch.compile(custom_op_func, backend=counting_backend, dynamic=True)

    for shape in shapes:
        cpu_input = torch.rand(shape, dtype=torch.bfloat16)
        npu_input = cpu_input.npu()
        dynamic_dim = dim if dim >= 0 else dim + len(shape)
        torch._dynamo.mark_dynamic(npu_input, dynamic_dim)  # pylint: disable=protected-access

        cpu_chunks = custom_op_func(cpu_input)
        npu_chunks = compiled_op(npu_input)
        _assert_chunk_outputs_equal(cpu_chunks, npu_chunks)

    assert compile_count == 1


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_torch_chunk_function_dynamic_shape_reuses_graph():
    """
    Feature: Test torch.chunk function call with dynamic chunk sizes
    Description: Cover call_function(torch.chunk), not only Tensor.chunk.
    Expectation: Outputs are correct and the compiled graph is reused.
    """
    compile_count = 0

    def counting_backend(gm, example_inputs):
        nonlocal compile_count
        compile_count += 1
        return backend(gm, example_inputs)

    def custom_op_func(input_tensor):
        return torch.chunk(input_tensor, 4, dim=0)

    compiled_op = torch.compile(custom_op_func, backend=counting_backend, dynamic=True)

    for shape in ([33, 16], [65, 16]):
        cpu_input = torch.rand(shape, dtype=torch.bfloat16)
        npu_input = cpu_input.npu()
        torch._dynamo.mark_dynamic(npu_input, 0)  # pylint: disable=protected-access

        cpu_chunks = custom_op_func(cpu_input)
        npu_chunks = compiled_op(npu_input)
        _assert_chunk_outputs_equal(cpu_chunks, npu_chunks)

    assert compile_count == 1
