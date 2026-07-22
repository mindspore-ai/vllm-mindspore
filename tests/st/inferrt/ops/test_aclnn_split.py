"""Tests for torch.split operation."""
import pytest
import torch
import torch_npu

from ms_inferrt.torch.fx_backend import backend as fx_backend
from ms_inferrt.torch.fx_mlir_backend import backend as mlir_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(input_self_tensor, split_size, dim=0):
    """Reference implementation of split."""
    return input_self_tensor.split(split_size, dim)


def get_op_func_compiled():
    """Get compiled split function."""
    def custom_op_func(input_self_tensor, split_size, dim=0):
        return input_self_tensor.split(split_size, dim)
    return torch.compile(custom_op_func, backend=mlir_backend)


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
    Description: Verify view -> split(int, dim=1) -> two linear branches matches NPU eager mode
    Expectation: Compiled NPU result matches NPU eager result
    """

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

    eager_out = func(input_tensor, weight)
    compiled_func = torch.compile(func, backend=fx_backend, fullgraph=True)
    compiled_out = compiled_func(input_tensor, weight)

    AssertRtolEqual(eager_out[0].detach().cpu(), compiled_out[0].detach().cpu())
    AssertRtolEqual(eager_out[1].detach().cpu(), compiled_out[1].detach().cpu())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_split_with_size_view_matches_eager():
    """
    Feature: Test split_with_size_view
    Description: Verify list split sizes lower to split_with_size_view and match CPU eager mode
    Expectation: Compiled NPU result matches CPU eager result
    """

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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_pangu_like_split_view_chain_matches_eager():
    """
    Feature: Test split/view/squeeze view chain from Pangu-like attention path
    Description: Verify split_with_size_view, view and squeeze_view keep eager semantics and produce contiguous final
                 output
    Expectation: Compiled NPU result matches CPU eager result and final output is contiguous
    """

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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_pangu_split_view_qkv_metadata_and_value_contiguous():
    """
    Feature: Locate Pangu split_with_size_view metadata before ATB consumers
    Description: Verify q/k/v split views keep eager strides and values, and v.contiguous() materializes value
    Expectation: Compiled metadata and values match eager mode
    """

    def func(out):
        q, k, v = torch.split(out, [1024, 128, 128], dim=-1)
        return q, k, v, v.contiguous()

    cpu_input = torch.randn(8, 1280, dtype=torch.float16)
    eager_out = func(cpu_input)
    compiled_func = torch.compile(func, backend=fx_backend, fullgraph=True)
    compiled_out = compiled_func(cpu_input.npu())

    for eager_tensor, compiled_tensor in zip(eager_out, compiled_out):
        AssertRtolEqual(eager_tensor.detach().cpu(), compiled_tensor.detach().cpu())
        assert tuple(compiled_tensor.shape) == tuple(eager_tensor.shape)
        assert tuple(compiled_tensor.stride()) == tuple(eager_tensor.stride())
        assert compiled_tensor.is_contiguous() == eager_tensor.is_contiguous()

    assert not compiled_out[0].is_contiguous()
    assert not compiled_out[1].is_contiguous()
    assert not compiled_out[2].is_contiguous()
    assert compiled_out[3].is_contiguous()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_pangu_split_view_rotary_ref_metadata_sync_matches_eager():
    """
    Feature: Validate ref metadata sync for apply_rotary_pos_emb after split_with_size_view
    Description: Use the key_rot branch, whose source split view carries a non-zero storage offset, and verify the
                 downstream squeeze/view chain matches eager mode without relying on exported tensor storage_offset
    Expectation: Compiled NPU result matches CPU eager result
    """

    def func(out, cos, sin):
        q, k, _ = torch.split(out, [1024, 128, 128], dim=-1)
        q = q.reshape(out.shape[0], 8, 128)
        k = k.reshape(out.shape[0], 1, 128)
        query_rot, key_rot = torch.ops.npu.npu_apply_rotary_pos_emb(
            q[None, ...],
            k[None, ...],
            cos[None, :, None, :],
            sin[None, :, None, :],
            "BSND",
        )
        query = query_rot.squeeze(0).view(-1, 8, 128)
        key = key_rot.squeeze(0).view(-1, 1, 128)
        return query, key

    cpu_out = torch.randn(8, 1280, dtype=torch.float16)
    cpu_cos = torch.randn(8, 128, dtype=torch.float16)
    cpu_sin = torch.randn(8, 128, dtype=torch.float16)

    eager_out = func(cpu_out.npu(), cpu_cos.npu(), cpu_sin.npu())
    compiled_func = torch.compile(func, backend=fx_backend, fullgraph=True)
    compiled_out = compiled_func(cpu_out.npu(), cpu_cos.npu(), cpu_sin.npu())

    for eager_tensor, compiled_tensor in zip(eager_out, compiled_out):
        AssertRtolEqual(eager_tensor.detach().cpu(), compiled_tensor.detach().cpu())
        assert tuple(compiled_tensor.shape) == tuple(eager_tensor.shape)
        assert tuple(compiled_tensor.stride()) == tuple(eager_tensor.stride())
        assert compiled_tensor.is_contiguous() == eager_tensor.is_contiguous()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_pangu_split_view_reshape_and_cache_requires_contiguous_value():
    """
    Feature: Locate ATB reshape_and_cache contiguous-input boundary after split_with_size_view
    Description: Verify value from split view must be materialized before entering InferRT ATB reshape_and_cache
    Expectation: Missing value.contiguous() raises the InferRT ATB contiguous-input error
    """

    def func(out, key_cache, value_cache, slot_indices):
        _, k, v = torch.split(out, [1024, 128, 128], dim=-1)
        key = k.contiguous().view(out.shape[0], 1, 128)
        value = v.view(out.shape[0], 1, 128)
        torch_npu._npu_reshape_and_cache(  # pylint: disable=protected-access
            key,
            value,
            key_cache,
            value_cache,
            slot_indices,
        )
        return key_cache, value_cache

    out = torch.randn(8, 1280, dtype=torch.float16).npu()
    key_cache = torch.zeros((2, 16, 1, 128), dtype=torch.float16).npu()
    value_cache = torch.zeros((2, 16, 1, 128), dtype=torch.float16).npu()
    slot_indices = torch.arange(8, dtype=torch.int32).npu()
    compiled_func = torch.compile(func, backend=fx_backend, fullgraph=True)

    with pytest.raises(RuntimeError, match="Only contiguous tensor is supported in atb now"):
        compiled_func(out, key_cache, value_cache, slot_indices)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_pangu_split_view_reshape_and_cache_with_contiguous_value_matches_eager():
    """
    Feature: Locate Pangu split_with_size_view plus ATB reshape_and_cache path
    Description: Verify materialized key/value from split view keep reshape_and_cache results aligned with eager mode
    Expectation: Eager and compiled cache updates match
    """

    def func(out, key_cache, value_cache, slot_indices, weight):
        _, k, v = torch.split(out, [1024, 128, 128], dim=-1)
        key = k.contiguous().view(out.shape[0], 1, 128)
        value = v.contiguous().view(out.shape[0], 1, 128)
        torch_npu._npu_reshape_and_cache(  # pylint: disable=protected-access
            key,
            value,
            key_cache,
            value_cache,
            slot_indices,
        )
        out_1 = torch.nn.functional.linear(value.view(out.shape[0], 128), weight, None)
        return key_cache, value_cache, out_1

    cpu_out = torch.randn(8, 1280, dtype=torch.float16)
    cpu_key_cache = torch.zeros((2, 16, 1, 128), dtype=torch.float16)
    cpu_value_cache = torch.zeros((2, 16, 1, 128), dtype=torch.float16)
    cpu_slot_indices = torch.arange(8, dtype=torch.int32)
    cpu_weight = torch.randn(4096, 128, dtype=torch.float16)

    eager_key_cache, eager_value_cache, eager_out = func(
        cpu_out.npu(),
        cpu_key_cache.npu(),
        cpu_value_cache.npu(),
        cpu_slot_indices.npu(),
        cpu_weight.npu(),
    )

    compiled_func = torch.compile(func, backend=fx_backend, fullgraph=True)
    compiled_key_cache, compiled_value_cache, compiled_out = compiled_func(
        cpu_out.npu(),
        cpu_key_cache.npu(),
        cpu_value_cache.npu(),
        cpu_slot_indices.npu(),
        cpu_weight.npu(),
    )

    AssertRtolEqual(eager_key_cache.detach().cpu(), compiled_key_cache.detach().cpu())
    AssertRtolEqual(eager_value_cache.detach().cpu(), compiled_value_cache.detach().cpu())
    AssertRtolEqual(eager_out.detach().cpu(), compiled_out.detach().cpu(), prec16=1e-2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_pangu_split_view_flash_attention_custom_call_matches_eager():
    """
    Feature: Locate Pangu split_with_size_view plus ATB flash_attention custom-call path
    Description: Verify materialized q/k/v from split view keep ATB flash_attention and following linear aligned
    Expectation: Eager and compiled outputs match
    """

    def func(out, mask, seq_len, weight):
        q, k, v = torch.split(out, [1024, 128, 128], dim=-1)
        query = q.contiguous().view(out.shape[0], 8, 128)
        key = k.contiguous().view(out.shape[0], 1, 128)
        value = v.contiguous().view(out.shape[0], 1, 128)
        flash_out = torch.empty_like(query)
        torch_npu._npu_flash_attention(  # pylint: disable=protected-access
            query,
            key,
            value,
            mask,
            seq_len,
            0.08838834764831843,
            8,
            1,
            flash_out,
        )
        attn_output = flash_out.view(out.shape[0], 1024)
        return torch.nn.functional.linear(attn_output, weight, None)

    cpu_out = torch.randn(8, 1280, dtype=torch.float16)
    cpu_mask = torch.zeros((8, 8), dtype=torch.float16)
    cpu_seq_len = torch.full((1,), 8, dtype=torch.int32)
    cpu_weight = torch.randn(4096, 1024, dtype=torch.float16)

    eager_out = func(cpu_out.npu(), cpu_mask.npu(), cpu_seq_len, cpu_weight.npu())
    compiled_func = torch.compile(func, backend=fx_backend, fullgraph=True)
    compiled_out = compiled_func(cpu_out.npu(), cpu_mask.npu(), cpu_seq_len, cpu_weight.npu())

    AssertRtolEqual(eager_out.detach().cpu(), compiled_out.detach().cpu(), prec16=1e-2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_pangu_split_view_v_contiguous_rotary_flash_attention_matches_eager():
    """
    Feature: Locate Pangu split_with_size_view path with only value materialized
    Description: Match network pattern: q/k stay as split views before rotary, only v is contiguous before ATB consumers
    Expectation: Eager and compiled outputs match
    """

    def func(out, cos, sin, mask, seq_len, weight):
        q, k, v = torch.split(out, [1024, 128, 128], dim=-1)
        v = v.contiguous()

        q = q.reshape(out.shape[0], 8, 128)
        k = k.reshape(out.shape[0], 1, 128)
        query_rot, key_rot = torch.ops.npu.npu_apply_rotary_pos_emb(
            q[None, ...],
            k[None, ...],
            cos[None, :, None, :],
            sin[None, :, None, :],
            "BSND",
        )

        query = query_rot.squeeze(0).view(-1, 8, 128)
        key = key_rot.squeeze(0).view(-1, 1, 128)
        value = v.view(-1, 1, 128)
        flash_out = torch.empty_like(query)
        torch_npu._npu_flash_attention(  # pylint: disable=protected-access
            query,
            key,
            value,
            mask,
            seq_len,
            0.08838834764831843,
            8,
            1,
            flash_out,
        )
        attn_output = flash_out.view(out.shape[0], 1024)
        return torch.nn.functional.linear(attn_output, weight, None)

    cpu_out = torch.randn(8, 1280, dtype=torch.float16)
    cpu_cos = torch.randn(8, 128, dtype=torch.float16)
    cpu_sin = torch.randn(8, 128, dtype=torch.float16)
    cpu_mask = torch.zeros((8, 8), dtype=torch.float16)
    cpu_seq_len = torch.full((1,), 8, dtype=torch.int32)
    cpu_weight = torch.randn(4096, 1024, dtype=torch.float16)

    eager_out = func(
        cpu_out.npu(),
        cpu_cos.npu(),
        cpu_sin.npu(),
        cpu_mask.npu(),
        cpu_seq_len,
        cpu_weight.npu(),
    )
    compiled_func = torch.compile(func, backend=fx_backend, fullgraph=True)
    compiled_out = compiled_func(
        cpu_out.npu(),
        cpu_cos.npu(),
        cpu_sin.npu(),
        cpu_mask.npu(),
        cpu_seq_len,
        cpu_weight.npu(),
    )

    AssertRtolEqual(eager_out.detach().cpu(), compiled_out.detach().cpu(), prec16=1e-2)
