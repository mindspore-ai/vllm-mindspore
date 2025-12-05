import math
import unittest
import numpy as np
import torch

import torch_npu
from mrt.torch import backend
from tests.mark_utils import arg_mark


def generate_int_tensor_with_sum(B, T):
    if B <= 0 or T < 0:
        raise ValueError("B 必须大于 0, T 必须是非负整数")
    if B > T:
        raise ValueError(f"无法生成 B={B} 个非负整数且总和为 T={T}, 每个数至少为 1, 所以 B <= T")
    
    partition_points = np.sort(np.random.choice(range(1, T), B - 1, replace = False))
    partition_points = np.concatenate([[0], partition_points, [T]])
    tensor = torch.tensor(np.diff(partition_points), dtype = torch.int)
    return tensor

def softmax(x):
    x = x.cpu().numpy().astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y
    return ans, x_sum, x_max

def compute_golden_output_cpu(query, key_cache, value_cache, num_heads, num_key_value_heads, head_dim, block_size, block_table, seq_lens, scale, input_layout = "TND"):
    B, H, D = query.shape
    assert H == num_heads, f"Query heads {H} != num_heads {num_heads}"
    assert D == head_dim, f"Query dim {D} != head_dim {head_dim}"

    Block_num, Block_size, total_dim = value_cache.shape
    value_head_dim = total_dim // num_key_value_heads

    block_num, block_size, _ = key_cache.shape
    T_kv = block_num * block_size

    key_cache_reshaped = key_cache.view(block_num, block_size, num_key_value_heads, head_dim)
    key_cache_reshaped = key_cache_reshaped.permute(2, 0, 1, 3)
    key_cache_reshaped = key_cache_reshaped.reshape(num_key_value_heads, T_kv, head_dim)

    value_cache_reshaped = value_cache.view(block_num, block_size, num_key_value_heads, value_head_dim)
    value_cache_reshaped = value_cache_reshaped.permute(2, 0, 1, 3)
    value_cache_reshaped = value_cache_reshaped.reshape(num_key_value_heads, T_kv, value_head_dim)

    if num_key_value_heads == 1 and num_heads > 1:
        key_cache_reshaped = key_cache_reshaped.expand(num_heads, T_kv, head_dim)
        value_cache_reshaped = value_cache_reshaped.expand(num_heads, T_kv, value_head_dim)
    elif num_key_value_heads == num_heads:
        pass
    else:
        raise NotImplementedError("不支持的 num_key_value_heads != num_heads 且 != 1 的情况")
    
    query_expanded = query.unsqueeze(-2)
    key_expanded = key_cache_reshaped.unsqueeze(0).expand(B, -1, -1, -1)
    value_expanded = value_cache_reshaped.unsqueeze(0).expand(B, -1, -1, -1)

    attn_scores = torch.matmul(query_expanded, key_expanded.transpose(-1, -2)) * scale
    mask = torch.arange(T_kv, device = query.device).unsqueeze(0) < seq_lens.unsqueeze(1)
    mask = mask.unsqueeze(1).unsqueeze(1)

    attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
    attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
    attention_out = torch.matmul(attn_weights, value_expanded)

    return attention_out


def supported_op_exec(query_states1, past_key, past_value, head_dim, B, N, S, softmax_lse_flag):
    attn_weights1 = torch.matmul(query_states1, past_key.transpose(2, 3)) / 0.0078125
    if (softmax_lse_flag == True):
        softmax_res, softmax_sum, softmax_max = softmax(attn_weights1)
        lse = np.log(softmax_sum) + softmax_max
    else:
        lse = np.zeros([B, N, S, 1], np.float32)
        attn_weights1 = torch.max(attn_weights1, torch.full(
            (1, 1), torch.finfo(attn_weights1.dtype).min, device=attn_weights1.device))
        attn_weights1 = torch.nn.functional.softmax(attn_weights1, dim=-1, dtype=torch.float32).to(query_states1.dtype)

    attn_output1 = torch.matmul(attn_weights1, past_value)
    return attn_output1, lse

def supported_op_exec_ntd(query_states1, past_key, past_value, head_dim, T, N):
    attn_weights1 = torch.matmul(query_states1, past_key.transpose(1, 2)) / 0.0078125

    attn_weights1 = torch.nn.functional.softmax(attn_weights1, dim=-1, dtype=torch.float32).to(query_states1.dtype)

    attn_output1 = torch.matmul(attn_weights1, past_value)
    attn_output1 = attn_output1.transpose(0, 1)
    return attn_output1

def custom_op_exec(query, key, value, head_dim, softmax_lse_flag):
    scale = 1 / 0.0078125
    return torch_npu.npu_fused_infer_attention_score(
        query, key, value, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)

def custom_op_exec_tnd(query, key, value, head_dim, softmax_lse_flag):
    scale = 1 / 0.0078125
    return torch_npu.npu_fused_infer_attention_score(
        query, key, value, num_heads=32, input_layout="TND", scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)

def custom_op_exec_ntd_tnd(query, key, value, head_dim, actseqlen, actseqlenkv, softmax_lse_flag):
    scale = 1 / 0.0078125
    return torch_npu.npu_fused_infer_attention_score(
        query, key, value, num_heads=2, input_layout="NTD_TND", scale=scale, pre_tokens=65535, next_tokens=65535, actual_seq_lengths=actseqlen, 
        actual_seq_lengths_kv=actseqlenkv, softmax_lse_flag=softmax_lse_flag)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_npu_fused_infer_attention_score():
    """
    Feature: Test aclnnFusedInferAttentionScore
    Description: base test
    Expectation: The result is correct
    """
    query = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
    key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
    value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()

    head_dim = 128

    golden_output, _ = custom_op_exec(query, key, value, head_dim, False)

    opt_fias_v3 = torch.compile(custom_op_exec, backend=backend)
    my_out, _ = opt_fias_v3(query, key, value, head_dim, False)
    assert np.allclose(golden_output.cpu(), my_out.cpu())


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_npu_fused_infer_attention_score_pfa_return_lse():
    """
    Feature: Test aclnnFusedInferAttentionScore
    Description: pfa and lse out
    Expectation: The result is correct
    """
    query = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
    key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
    value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()

    head_dim = 128
    softmax_lse_flag = True

    golden_out_attention, golden_out_lse = custom_op_exec(query, key, value, head_dim, softmax_lse_flag)

    opt_fias_v3 = torch.compile(custom_op_exec, backend=backend)
    my_out_attention, my_out_lse = opt_fias_v3(query, key, value, head_dim, softmax_lse_flag)

    assert np.allclose(golden_out_attention.cpu(), my_out_attention.cpu())
    assert np.allclose(golden_out_lse.cpu(), my_out_lse.cpu())


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_npu_fused_infer_attention_score_ifa_return_lse():
    """
    Feature: Test aclnnFusedInferAttentionScore
    Description: ifa and lse out
    Expectation: The result is correct
    """
    query = torch.randn(1, 32, 1, 128, dtype=torch.float16).npu()
    key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
    value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()

    head_dim = 128
    softmax_lse_flag = True

    golden_out_attention, golden_out_lse = custom_op_exec(query, key, value, head_dim, softmax_lse_flag)

    opt_fias_v3 = torch.compile(custom_op_exec, backend=backend)
    my_out_attention, my_out_lse = opt_fias_v3(query, key, value, head_dim, softmax_lse_flag)
    assert np.allclose(golden_out_attention.cpu(), my_out_attention.cpu())
    assert np.allclose(golden_out_lse.cpu(), my_out_lse.cpu())


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_npu_fused_infer_attention_score_ntd_tnd():
    """
    Feature: Test aclnnFusedInferAttentionScore
    Description: ntd_tnd
    Expectation: The result is correct
    """
    query = torch.full((2, 32, 192), 1, dtype=torch.bfloat16).npu()
    key = torch.full((2, 32, 192), 1, dtype=torch.bfloat16).npu()
    value = torch.full((2, 32, 128), 1, dtype=torch.bfloat16).npu()

    head_dim = 128
    softmax_lse_flag = False

    actseqlen = [32]
    actseqlenkv = [32]

    golden_output, _ = custom_op_exec_ntd_tnd(query, key, value, head_dim, actseqlen, actseqlenkv, softmax_lse_flag)

    opt_fias_v3 = torch.compile(custom_op_exec_ntd_tnd, backend=backend)
    my_out, _ = opt_fias_v3(query, key, value, head_dim, actseqlen, actseqlenkv, softmax_lse_flag)
    assert np.allclose(golden_output.to(torch.float16).cpu(), my_out.to(torch.float16).cpu())


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_npu_fused_infer_attention_score_v3_fullquant():
    """
    Feature: Test aclnnFusedInferAttentionScore
    Description: fullquant
    Expectation: The result is correct
    """
    query = torch.randint(1, 2, (1, 32, 2, 128), dtype=torch.int8).npu()
    key = torch.randint(1, 2, (1, 32, 2048, 128), dtype=torch.int8).npu()
    value = torch.randint(1, 2, (1, 32, 2048, 128), dtype=torch.int8).npu()
    dequant_scale1 = torch.ones(1).npu()
    dequant_scale2 = torch.ones(1).npu()
    quant_scale1 = torch.ones(1).npu()
    quant_scale2 = torch.ones(1).npu()

    head_dim = 128
    softmax_lse_flag = False
    scale = 1 / 0.0078125

    def fias_v3_fullquant(query, key, value, dequant_scale1, dequant_scale2, quant_scale1,
        quant_scale2, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535, next_tokens=65535,
        softmax_lse_flag=False):
        return torch_npu.npu_fused_infer_attention_score(
            query, key, value, dequant_scale1=dequant_scale1, dequant_scale2=dequant_scale2, quant_scale1=quant_scale1,
            quant_scale2=quant_scale2, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535, next_tokens=65535,
            softmax_lse_flag=softmax_lse_flag)
    
    golden_output, _ = fias_v3_fullquant(query, key, value, dequant_scale1=dequant_scale1, dequant_scale2=dequant_scale2, quant_scale1=quant_scale1,
            quant_scale2=quant_scale2, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535, next_tokens=65535,
            softmax_lse_flag=softmax_lse_flag)
    
    opt_fias_v3 = torch.compile(fias_v3_fullquant, backend=backend)
    my_out, _ = opt_fias_v3(query, key, value, dequant_scale1=dequant_scale1, dequant_scale2=dequant_scale2, quant_scale1=quant_scale1,
            quant_scale2=quant_scale2, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535, next_tokens=65535,
            softmax_lse_flag=softmax_lse_flag)
    res = my_out.equal(golden_output)
    assert res


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_npu_infer_attention_score_PA():
    """
    Feature: Test aclnnFusedInferAttentionScore
    Description: PA
    Expectation: The result is correct
    """
    B = 1
    T = 69
    head_dim = 128
    num_heads = 16
    num_kv_heads = 1
    block_num = 1
    block_size = 128
    block_table = torch.randint(0, 10, [B, 32], dtype = torch.int32).npu()
    query = torch.rand([B * 1, num_heads, head_dim], dtype=torch.float16).npu()
    key_cache = torch.rand([block_num, block_size, num_kv_heads * head_dim], dtype=torch.float16).npu()
    value_cache = torch.rand([block_num, block_size, num_kv_heads * 128], dtype=torch.float16).npu()
    scale = head_dim**-0.5

    seq_lens = generate_int_tensor_with_sum(B, T).to(torch.int32)
    query_lens = torch.ones(B, dtype = torch.int32).npu()

    def fias_v3_pa(query, key_cache, value_cache, num_heads, num_kv_heads, input_layout,
                        scale, block_table, block_size, query_lens, seq_lens):
        return torch_npu.npu_fused_infer_attention_score(query, key_cache, value_cache, num_heads = num_heads, num_key_value_heads = num_kv_heads, input_layout = input_layout,
                        scale = scale, block_table = block_table, block_size = block_size, actual_seq_lengths = query_lens, actual_seq_lengths_kv = seq_lens)[0]
    
    golden_output = fias_v3_pa(query, key_cache, value_cache, num_heads, num_kv_heads, "TND",
                        scale, block_table, block_size, query_lens, seq_lens)

    opt_fias_v3 = torch.compile(fias_v3_pa, backend=backend)
    my_out = opt_fias_v3(query, key_cache, value_cache, num_heads, num_kv_heads, "TND",
                        scale, block_table, block_size, query_lens, seq_lens)

    assert np.allclose(golden_output.cpu(), my_out.cpu(), equal_nan=True)


def run_npu_fused_infer_attention_score(query, key, value, key_antiquant_scale, value_antiquant_scale,
                key_antiquant_mode, value_antiquant_mode, num_heads=8, input_layout="BNSD", 
                scale=0.0, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=False):
    return torch_npu.npu_fused_infer_attention_score(
                query, key, value, key_antiquant_scale=key_antiquant_scale, value_antiquant_scale=value_antiquant_scale,
                key_antiquant_mode=key_antiquant_mode, value_antiquant_mode=value_antiquant_mode, num_heads=num_heads, input_layout=input_layout, 
                scale=scale, pre_tokens=pre_tokens, next_tokens=next_tokens, softmax_lse_flag=softmax_lse_flag)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_npu_fused_infer_attention_score_v3_antiquant():
    """
    Feature: Test aclnnFusedInferAttentionScore
    Description: antiquant
    Expectation: The result is correct
    """
    query = torch.ones(32, 8, 1, 128, dtype=torch.float16).npu()
    key = torch.full((32, 8, 2048, 16), 286331353, dtype=torch.int32).npu()
    value = torch.full((32, 8, 2048, 16), 286331353, dtype=torch.int32).npu()
    key_antiquant_scale = torch.ones(1, 32, 2048, dtype=torch.float32).npu()
    value_antiquant_scale = torch.ones(1, 32, 2048, dtype=torch.float32).npu()
    key_antiquant_mode = torch.ones(1).npu()
    value_antiquant_mode = torch.ones(1).npu()

    softmax_lse_flag = False
    scale = 1 / 0.0078125

    opt_fias_v3 = torch.compile(run_npu_fused_infer_attention_score, backend=backend)

    attention_out_std, _ = run_npu_fused_infer_attention_score(query, key, value, key_antiquant_scale=key_antiquant_scale, value_antiquant_scale=value_antiquant_scale,
        key_antiquant_mode=key_antiquant_mode, value_antiquant_mode=value_antiquant_mode, num_heads=8, input_layout="BNSD", 
        scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)
    attention_out, _ = opt_fias_v3(query, key, value, key_antiquant_scale=key_antiquant_scale, value_antiquant_scale=value_antiquant_scale,
        key_antiquant_mode=key_antiquant_mode, value_antiquant_mode=value_antiquant_mode, num_heads=8, input_layout="BNSD", 
        scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)
    assert attention_out.equal(attention_out_std)
