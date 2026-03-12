"""Tests for torch.ops.npu.npu_moe_token_unpermute operation."""
import pytest
import torch

from ms_inferrt.torch.fx_mlir_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def moe_token_unpermute_with_padding(
    tokens: torch.Tensor,
    indices: torch.Tensor,
    probs: torch.Tensor,
    original_shape: torch.Size,
) -> torch.Tensor:
    """Unpermute tokens with padding support."""
    assert tokens.dim() == 2, f"Got {tokens.dims()} D"

    probs = probs.view(-1).unsqueeze(-1)
    indices = indices.view(-1, 1).expand(-1, tokens.shape[1])
    assert (tokens.shape == indices.shape), "Shape mismatch between tokens and indices."
    merged_out = probs * tokens
    zero_tensor = torch.zeros(original_shape, dtype=merged_out.dtype, device=merged_out.device)
    org_tokens = torch.scatter_add(zero_tensor, 0, indices, merged_out)
    return org_tokens


def moe_token_unpermute_golden(
    tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
    padded_mode: bool = False,
    original_shape: torch.Size = None,
):
    """Reference implementation of moe_token_unpermute."""
    if padded_mode:
        return moe_token_unpermute_with_padding(tokens, sorted_indices, probs, original_shape)

    assert tokens.size(0) == sorted_indices.numel()
    num_org_tokens, topk = (tokens.size(0), 1) if probs is None else (probs.numel(), probs.size(1))

    org_tokens = torch.zeros(
        [num_org_tokens, tokens.shape[-1]],
        dtype=tokens.dtype,
        device=tokens.device,
    )
    org_tokens.index_copy_(0, sorted_indices, tokens)
    org_tokens = org_tokens.reshape(-1, topk, tokens.size(-1))
    if probs is not None:
        org_tokens = org_tokens * probs.unsqueeze(-1)
    org_tokens = org_tokens.sum(dim=1)
    return org_tokens


def op_func(permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
    padded_mode: bool = False,
    restore_shape: torch.Size = None,
):
    """Reference op function for moe_token_unpermute."""
    return moe_token_unpermute_golden(permuted_tokens, sorted_indices, probs, padded_mode, restore_shape)


def get_op_func_compiled():
    """Get compiled moe_token_unpermute function."""
    def custom_op_func(permuted_tokens, sorted_indices, probs=None, padded_mode=False, restore_shape=None):
        return torch.ops.npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs, padded_mode, restore_shape)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("num_tokens", [1024, 2048])
@pytest.mark.parametrize("hidden_size", [6144, 8192])
@pytest.mark.parametrize("topk", [1, 4])
@pytest.mark.parametrize("num_experts", [4, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_token_unpermute(num_tokens, hidden_size, topk, num_experts, dtype):
    """
    Feature: Test aclnn moe_token_unpermute
    Description: Test aclnn moe_token_unpermute with fp16/bf16 inputs
    Expectation: The result is correct
    """

    permute_token_cpu = torch.randn(num_tokens * topk, hidden_size).to(dtype)
    indices_cpu = torch.randint(0, num_experts, (num_tokens, topk))
    sorted_indices_cpu = torch.argsort(indices_cpu.view(-1), stable=True).to(dtype=torch.int32)
    probs_cpu = None
    probs_npu = None
    if topk > 1:
        probs_cpu = (torch.ones(num_tokens, topk) / topk).to(dtype).requires_grad_(True)
        probs_npu = probs_cpu.clone().npu().requires_grad_(True)

    permute_token_npu = permute_token_cpu.clone().npu().requires_grad_(True)
    sorted_indices_npu = sorted_indices_cpu.clone().npu()

    sorted_indices_cpu = torch.argsort(sorted_indices_cpu, stable=True)
    cpu_output0 = op_func(permute_token_cpu, sorted_indices_cpu, probs=probs_cpu)

    op_func_compiled = get_op_func_compiled()
    npu_output0 = op_func_compiled(permute_token_npu, sorted_indices_npu, probs=probs_npu).detach().cpu()
    AssertRtolEqual(cpu_output0, npu_output0)
