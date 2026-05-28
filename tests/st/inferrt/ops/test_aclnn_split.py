"""Tests for torch.split operation."""
import os
from pathlib import Path

import pytest
import torch

from ms_inferrt.torch.fx_backend import backend as fx_backend
from ms_inferrt.torch.fx_mlir_backend import backend as mlir_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual

_IR_DUMP_ENV = "MS_INFERRT_DEV_DUMP_IR"


def op_func(input_self_tensor, split_size, dim=0):
    """Reference implementation of split."""
    return input_self_tensor.split(split_size, dim)


def get_op_func_compiled():
    """Get compiled split function."""
    def custom_op_func(input_self_tensor, split_size, dim=0):
        return input_self_tensor.split(split_size, dim)
    return torch.compile(custom_op_func, backend=mlir_backend)


def _dumped_ir_path() -> Path:
    return Path(f"graph_{os.getpid()}.txt")


def _reset_ir_dump():
    if os.environ.get(_IR_DUMP_ENV, "") != "1":
        return
    ir_path = _dumped_ir_path()
    if ir_path.exists():
        ir_path.unlink()
    torch.compiler.reset()


def _assert_dumped_ir_contains(expected: str):
    if os.environ.get(_IR_DUMP_ENV, "") != "1":
        return
    ir_path = _dumped_ir_path()
    assert ir_path.exists(), f"missing IR dump: {ir_path}"
    ir_text = ir_path.read_text()
    assert expected in ir_text, f"{expected} not found in {ir_path}"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [[128, 4096], [32, 1024]])
def test_split_tensor(shape):
    """
    Feature: Test split
    Description: Test split with bf16 inputs
    Expectation: The result is correct
    """

    self_tensor = torch.rand(shape, dtype=torch.bfloat16)
    dim = 0
    self_tensor_npu = self_tensor.npu()
    split_size = shape[0] // 2

    cpu_output0, cpu_output1 = op_func(self_tensor, split_size, dim=dim)
    op_func_compiled = get_op_func_compiled()
    npu_output0, npu_output1 = op_func_compiled(self_tensor_npu, split_size, dim=dim)
    npu_output_opt0 = npu_output0.detach().cpu()
    npu_output_opt1 = npu_output1.detach().cpu()
    AssertRtolEqual(cpu_output0, npu_output_opt0)
    AssertRtolEqual(cpu_output1, npu_output_opt1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("shape", [[128, 4096], [32, 1024]])
def test_split_with_size(shape):
    """
    Feature: Test split
    Description: Test split with bf16 inputs
    Expectation: The result is correct
    """

    self_tensor = torch.rand(shape, dtype=torch.bfloat16)
    dim = 0
    self_tensor_npu = self_tensor.npu()
    split_size = [shape[0] // 2, shape[0] // 2]

    cpu_output0, cpu_output1 = op_func(self_tensor, split_size, dim=dim)
    op_func_compiled = get_op_func_compiled()
    npu_output0, npu_output1 = op_func_compiled(self_tensor_npu, split_size, dim=dim)
    npu_output_opt0 = npu_output0.detach().cpu()
    npu_output_opt1 = npu_output1.detach().cpu()
    AssertRtolEqual(cpu_output0, npu_output_opt0)
    AssertRtolEqual(cpu_output1, npu_output_opt1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("seq_len", [8], ids=["even_seq"])
def test_split_tensor_view_deepseek_indexer_pattern(seq_len):
    """
    Feature: Test split_tensor_view in DeepSeek indexer-like pattern
    Description: Verify view -> split(int, dim=1) -> two linear branches matches CPU eager mode
    Expectation: Compiled NPU result matches CPU eager result and IR contains split_tensor_view when IR dump is enabled
    """

    _reset_ir_dump()

    def func(input_tensor, weight):
        x = input_tensor.view(1, -1, 6144)
        split_size = x.size(1) // 2
        x_prev, x_next = torch.functional.split(x, split_size, dim=1)
        weight_prev = torch.nn.functional.linear(x_prev, weight, None)
        weight_next = torch.nn.functional.linear(x_next, weight, None)
        return weight_prev, weight_next

    cpu_input = torch.randn(seq_len, 6144, dtype=torch.bfloat16)
    cpu_weight = torch.randn(32, 6144, dtype=torch.bfloat16)
    input_tensor = cpu_input.npu()
    weight = cpu_weight.npu()

    eager_out = func(cpu_input, cpu_weight)
    compiled_func = torch.compile(func, backend=fx_backend, fullgraph=True)
    compiled_out = compiled_func(input_tensor, weight)

    AssertRtolEqual(eager_out[0].detach().cpu(), compiled_out[0].detach().cpu())
    AssertRtolEqual(eager_out[1].detach().cpu(), compiled_out[1].detach().cpu())
    _assert_dumped_ir_contains("ops.split_tensor_view")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_split_with_size_view_matches_eager():
    """
    Feature: Test split_with_size_view
    Description: Verify list split sizes lower to split_with_size_view and match CPU eager mode
    Expectation: Compiled NPU result matches CPU eager result and IR contains split_with_size_view when IR dump is enabled
    """

    _reset_ir_dump()

    def func(input_tensor):
        x = input_tensor.view(1, -1, 6144)
        x_prev, x_next = torch.split(x, [4, 5], dim=1)
        return x_prev + 1, x_next - 1

    cpu_input = torch.randn(9, 6144, dtype=torch.bfloat16)
    input_tensor = cpu_input.npu()
    eager_out = func(cpu_input)
    compiled_func = torch.compile(func, backend=fx_backend, fullgraph=True)
    compiled_out = compiled_func(input_tensor)

    AssertRtolEqual(eager_out[0].detach().cpu(), compiled_out[0].detach().cpu())
    AssertRtolEqual(eager_out[1].detach().cpu(), compiled_out[1].detach().cpu())
    _assert_dumped_ir_contains("ops.split_with_size_view")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_pangu_like_split_view_chain_matches_eager():
    """
    Feature: Test split/view/squeeze view chain from Pangu-like attention path
    Description: Verify split_with_size_view, view and squeeze_view keep eager semantics and produce contiguous final output
    Expectation: Compiled NPU result matches CPU eager result, final output is contiguous, and IR contains the expected view ops
    """

    _reset_ir_dump()

    hidden_size = 4096
    num_heads = 8
    head_dim = 128
    q_size = num_heads * head_dim
    kv_size = head_dim

    def func(hidden_states, qkv_weight, o_proj_weight):
        out = torch.nn.functional.linear(hidden_states, qkv_weight, None)
        q, k, v = torch.split(out, [q_size, kv_size, kv_size], dim=-1)

        q = q.reshape(hidden_states.shape[0], num_heads, head_dim)
        k = k.reshape(hidden_states.shape[0], 1, head_dim)
        v = v.view(hidden_states.shape[0], 1, head_dim)

        query_rot = q.view(1, hidden_states.shape[0], num_heads, head_dim)
        key_rot = k.view(1, hidden_states.shape[0], 1, head_dim)
        value_rot = v.view(1, hidden_states.shape[0], 1, head_dim)

        query = query_rot.squeeze(0)
        key = key_rot.squeeze(0)
        value = value_rot.squeeze(0)

        attn_output = query.view(hidden_states.shape[0], q_size)
        out_1 = torch.nn.functional.linear(attn_output, o_proj_weight, None)
        out_1 = out_1 + (key.sum() + value.sum()) * 0
        return out_1

    cpu_input = torch.randn(16, hidden_size, dtype=torch.float16)
    cpu_qkv_weight = torch.randn(q_size + 2 * kv_size, hidden_size, dtype=torch.float16)
    cpu_o_proj_weight = torch.randn(hidden_size, q_size, dtype=torch.float16)

    hidden_states = cpu_input.npu()
    qkv_weight = cpu_qkv_weight.npu()
    o_proj_weight = cpu_o_proj_weight.npu()

    eager_out = func(cpu_input, cpu_qkv_weight, cpu_o_proj_weight)
    compiled_func = torch.compile(func, backend=fx_backend, fullgraph=True)
    compiled_out = compiled_func(hidden_states, qkv_weight, o_proj_weight)

    assert tuple(compiled_out.shape) == tuple(eager_out.shape)
    assert compiled_out.is_contiguous() == eager_out.is_contiguous()
    AssertRtolEqual(
        eager_out.detach().cpu(),
        compiled_out.detach().cpu(),
        prec16=1e-2,
    )
    _assert_dumped_ir_contains("ops.split_with_size_view")
    _assert_dumped_ir_contains("ops.squeeze_view")
    _assert_dumped_ir_contains("ops.view")
