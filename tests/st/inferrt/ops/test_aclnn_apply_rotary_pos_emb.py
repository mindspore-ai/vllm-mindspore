import pytest
import numpy as np
import torch

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual
from mrt.torch.fx_mlir_backend import backend


def op_func(query, key, cos, sin):
    x1 = query[..., :query.shape[-1] // 2]
    x2 = query[..., query.shape[-1] // 2 :]
    concat = torch.cat([-x2, x1], dim=-1)
    x2_mul = concat * sin
    x1_mul = query * cos
    res0 = x2_mul + x1_mul

    k1 = key[..., :key.shape[-1] // 2]
    k2 = key[..., key.shape[-1] // 2:]
    concatk = torch.cat([-k2, k1], dim=-1)
    x1k_mul = concatk * sin
    x2k_mul = key * cos
    res1 = x2k_mul + x1k_mul
    return res0, res1


def get_op_func_compiled():
    def custom_op_func(query, key, cos, sin):
        return torch.ops.npu.npu_apply_rotary_pos_emb(query, key, cos, sin, '1')
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("num_heads", [16, 32])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_apply_rotary_pos_emb(pipeline, monkeypatch, batch_size, num_heads, dtype):
    """
    Feature: Test aclnn apply_rotary_pos_emb
    Description: Test aclnn apply_rotary_pos_emb with fp16/bf16 inputs
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    query_shape = [batch_size, 1024, num_heads, 128]
    cos_shape = [batch_size, 1024, 1, 128]

    query_cpu = torch.rand(query_shape, dtype=dtype)
    key_cpu = torch.rand(query_shape, dtype=dtype)
    cos_cpu = torch.rand(cos_shape, dtype=dtype)
    sin_cpu = torch.rand(cos_shape, dtype=dtype)

    query_npu = query_cpu.clone().npu()
    key_npu = key_cpu.clone().npu()
    cos_npu = cos_cpu.clone().npu()
    sin_npu = sin_cpu.clone().npu()

    cpu_output0, cpu_output1 = op_func(query_cpu, key_cpu, cos_cpu, sin_cpu)
    op_func_compiled = get_op_func_compiled()
    npu_output0, npu_output1 = op_func_compiled(query_npu, key_npu, cos_npu, sin_npu)
    AssertRtolEqual(cpu_output0, npu_output0.detach().cpu())
    AssertRtolEqual(cpu_output1, npu_output1.detach().cpu())
