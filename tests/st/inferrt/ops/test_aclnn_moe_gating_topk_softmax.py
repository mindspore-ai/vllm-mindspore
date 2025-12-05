import pytest
import numpy as np
import torch

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual
from mrt.torch.fx_mlir_backend import backend



def softmax_func(x):
    is_fp16 = x.dtype == torch.float16
    x = x.to(torch.float32)
    x_max = x.max(axis=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)

    zero_mask = (x_sum == 0)
    ans = torch.where(zero_mask, torch.tensor(0.0, device=x.device), y / x_sum)
    if is_fp16:
        ans = ans.to(torch.float16)
        x_max = x_max.to(torch.float16)
        x_sum = x_sum.to(torch.float16)
    return ans, x_max, x_sum

def op_func(x, finished_optional, k):
    num_expert = x.shape[-1]
    softmax, _, _ = softmax_func(x)

    _, expert_idx = torch.sort(softmax, dim=-1, descending=True, stable=True)
    expert_idx = expert_idx[:,:k]
    y = torch.gather(softmax, -1, expert_idx)

    if finished_optional is not None:
        finished_optional = finished_optional.reshape(finished_optional.shape[0], 1)
        finished_optional = finished_optional.repeat(1, k)
        expert_idx = torch.where(finished_optional, num_expert, expert_idx)
    
    batch_size, k_size = y.shape[0], y.shape[1]
    row_idx = torch.arange(batch_size * k_size, device=x.device).reshape(k_size, batch_size).t()

    if x.dtype == torch.float16:
        y = y.to(torch.float16)

    return y, expert_idx.to(torch.int32), row_idx.to(torch.int32)



def get_op_func_compiled():
    def custom_op_func(x, finished, k):
        return torch.ops.npu.npu_moe_gating_top_k_softmax(x, finished, k)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
@pytest.mark.parametrize("n", [10, 420, 520])
@pytest.mark.parametrize("k", [2, 4, 5, 9])
@pytest.mark.parametrize("col", [200, 1256, 5120])
def test_moe_gating_topk_softmax(pipeline, monkeypatch, dtype, n, k, col):
    """
    Feature: Test aclnn moe_gating_topk_softmax
    Description: Test aclnn moe_gating_topk_softmax with fp32/fp16 inputs
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")

    x = torch.rand(n, col, dtype=dtype)
    finished = torch.rand(n).to(torch.bool)

    x_npu = x.npu()
    finished_npu =  finished.npu()

    y_golden, expert_idx_golden, row_idx_golden = op_func(x, finished, k)
    op_func_compiled = get_op_func_compiled()
    y, expert_idx, row_idx = [npu_output.detach().cpu() for npu_output in op_func_compiled(x_npu, finished_npu, k)]
    AssertRtolEqual(y, y_golden)
    AssertRtolEqual(expert_idx, expert_idx_golden)
    AssertRtolEqual(row_idx, row_idx_golden)
