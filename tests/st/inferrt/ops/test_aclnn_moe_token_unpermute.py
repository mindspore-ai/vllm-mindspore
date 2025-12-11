import pytest
import numpy as np
import torch

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual
from mrt.torch.fx_mlir_backend import backend


def moe_token_unpermute_with_padding(
    permuted_tokens: torch.Tensor,
    indices: torch.Tensor,
    probs: torch.Tensor,
    restore_shape: torch.Size, 
)->torch.Tensor:
    assert permuted_tokens.dim() == 2, f"Got {permuted_tokens.dims()} D"
    # Reshape and expand probabilities and indices to match permuted_tokens
    probs = probs.view(-1).unsqueeze(-1)
    indices = indices.view(-1, 1).expand(-1, permuted_tokens.shape[1])
    assert (
        permuted_tokens.shape == indices.shape
    ), "Shape mismatch between permuted_tokens and indices."

    # Combine tokens with their probabilities
    combined_output = probs * permuted_tokens

    # Prepare a tensor of zeros with the desired output shape
    empty_tokens = torch.zeros(
        restore_shape,
        dtype=combined_output.dtype,
        device=combined_output.device,
    )

    # Scatter the combined tokens back to their original positions
    unpermuted_tokens = torch.scatter_add(empty_tokens, 0, indices, combined_output)
    return unpermuted_tokens

def moe_token_unpermute_golden(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
    padded_mode: bool = False,
    restore_shape: torch.Size = None
):
    if padded_mode:
        return moe_token_unpermute_with_padding(permuted_tokens, sorted_indices, probs, restore_shape)

    assert permuted_tokens.size(0) == sorted_indices.numel()
    if probs is not None:
        num_unpermute_tokens = probs.numel()
        topk = probs.size(1)
    else:
        num_unpermute_tokens = permuted_tokens.size(0)
        topk = 1
    
    unpermuted_tokens = torch.zeros(
        [num_unpermute_tokens, permuted_tokens.shape[-1]],
        dtype = permuted_tokens.dtype,
        device = permuted_tokens.device,
    )
    unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)
    return unpermuted_tokens
    

def op_func(permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
    padded_mode: bool = False,
    restore_shape: torch.Size = None
):
    return moe_token_unpermute_golden(permuted_tokens, sorted_indices, probs, padded_mode, restore_shape)


def get_op_func_compiled():
    def custom_op_func(permuted_tokens, sorted_indices, probs = None,  padded_mode = False, restore_shape = None):
        return torch.ops.npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs, padded_mode, restore_shape)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
@pytest.mark.parametrize("num_tokens", [1024, 2048])
@pytest.mark.parametrize("hidden_size", [6144, 8192])
@pytest.mark.parametrize("topk", [1, 4])
@pytest.mark.parametrize("num_experts", [4, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_token_unpermute(pipeline, monkeypatch, num_tokens, hidden_size, topk, num_experts, dtype):
    """
    Feature: Test aclnn moe_token_unpermute
    Description: Test aclnn moe_token_unpermute with fp16/bf16 inputs
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")


    permute_token_cpu = torch.randn(num_tokens * topk, hidden_size).to(dtype)
    indices_cpu = torch.randint(0, num_experts, (num_tokens, topk))
    sorted_indices_cpu = torch.argsort(indices_cpu.view(-1), stable=True).to(dtype=torch.int32)
    probs_cpu = None
    probs_npu = None
    if topk > 1:
        probs_cpu = (torch.ones(num_tokens, topk)/topk).to(dtype).requires_grad_(True)
        probs_npu = probs_cpu.clone().npu().requires_grad_(True)
    
    permute_token_npu = permute_token_cpu.clone().npu().requires_grad_(True)
    sorted_indices_npu = sorted_indices_cpu.clone().npu()

    sorted_indices_cpu = torch.argsort(sorted_indices_cpu, stable=True)
    cpu_output0 = op_func(permute_token_cpu, sorted_indices_cpu, probs=probs_cpu)

    op_func_compiled = get_op_func_compiled()
    npu_output0 =  op_func_compiled(permute_token_npu, sorted_indices_npu, probs=probs_npu).detach().cpu()
    AssertRtolEqual(cpu_output0, npu_output0)
