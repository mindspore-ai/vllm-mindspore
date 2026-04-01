"""Tests for torch.ops.npu.npu_moe_re_routing operation."""
import math
import random

import pytest
import torch
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from ms_inferrt.torch.fx_mlir_backend import backend as mlir_backend
from ms_inferrt.torch.fx_backend import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def build_inputs(tokens_num=1024, tokens_length=256, rank_num=4, expert_num=4):
    """Build random inputs for npu_moe_re_routing tests."""
    tokens = torch.randint(low=-10, high=20, size=(tokens_num, tokens_length), dtype=torch.int8)
    expert_token_num_per_rank = torch.ones(rank_num, expert_num, dtype=torch.int32)

    tokens_sum = 0
    for i in range(rank_num):
        for j in range(expert_num):
            if i == rank_num - 1 and j == expert_num - 1:
                expert_token_num_per_rank[i][j] = tokens_num - tokens_sum
                break
            if tokens_num >= rank_num * expert_num:
                floor = max(1, math.floor(tokens_num / (rank_num * expert_num)))
                rand_num = random.randint(1, floor)
            elif tokens_sum >= tokens_num:
                rand_num = 0
            else:
                rand_int = tokens_num - tokens_sum
                rand_num = random.randint(0, max(rand_int, 1))
            rand_num = min(rand_num, max(tokens_num - tokens_sum, 0))
            expert_token_num_per_rank[i][j] = rand_num
            tokens_sum += rand_num

    per_token_scales = torch.randn(tokens_num, dtype=torch.float32)
    return tokens, expert_token_num_per_rank, per_token_scales


class MoeReRoutingModel(torch.nn.Module):
    def forward(self, tokens, expert_token_num_per_rank, per_token_scales=None, expert_token_num_type=1, idx_type=0):
        return torch.ops.npu.npu_moe_re_routing(
            tokens,
            expert_token_num_per_rank,
            per_token_scales=per_token_scales,
            expert_token_num_type=expert_token_num_type,
            idx_type=idx_type,
        )


def _run_eager_and_compiled(tokens, expert_token_num_per_rank, per_token_scales,
                            expert_token_num_type, idx_type, backend):
    """Run MoeReRoutingModel in eager and compiled modes and return both outputs."""
    model = MoeReRoutingModel().npu()
    model.eval()

    tokens_npu = tokens.npu()
    expert_token_num_per_rank_npu = expert_token_num_per_rank.npu()
    per_token_scales_npu = per_token_scales.npu()

    with torch.no_grad():
        eager_out = model(tokens_npu, expert_token_num_per_rank_npu, per_token_scales_npu, expert_token_num_type,
                          idx_type)

    compiled_model = torch.compile(model, backend=backend, dynamic=True, fullgraph=False)
    with torch.no_grad():
        compiled_out = compiled_model(
            tokens_npu, expert_token_num_per_rank_npu, per_token_scales_npu, expert_token_num_type, idx_type
        )

    eager_out = [t.detach().cpu() for t in eager_out]
    compiled_out = [t.detach().cpu() for t in compiled_out]
    return eager_out, compiled_out


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend", (fx_backend, mlir_backend))
@pytest.mark.parametrize("tokens_num,tokens_length,rank_num,expert_num", [(512, 128, 4, 4), (1024, 256, 8, 8)])
def test_aclnn_moe_re_routing(backend, tokens_num, tokens_length, rank_num, expert_num):
    """
    Feature: Test aclnnMoeReRouting
    Description: Compare eager mode and compiled mode results for npu_moe_re_routing
    Expectation: The results are close
    """
    torch.manual_seed(0)
    random.seed(0)

    tokens, expert_token_num_per_rank, per_token_scales = build_inputs(
        tokens_num=tokens_num,
        tokens_length=tokens_length,
        rank_num=rank_num,
        expert_num=expert_num,
    )
    expert_token_num_type = 1
    idx_type = 0

    eager_out, compiled_out = _run_eager_and_compiled(
        tokens, expert_token_num_per_rank, per_token_scales, expert_token_num_type, idx_type, backend
    )

    for eager_tensor, compiled_tensor in zip(eager_out, compiled_out):
        AssertRtolEqual(eager_tensor, compiled_tensor)
