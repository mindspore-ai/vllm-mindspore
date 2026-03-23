"""
Tests for graph optimization with redundant copy elimination.

This test verifies that copy_elimination pass correctly removes
unnecessary copy_ operations from FX graph when the destination
is neither a graph input nor output.
"""
import os

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from ms_inferrt.torch.fx_backend import backend as fx_backend

from tests.mark_utils import arg_mark


class LinearAllReduceModule(nn.Module):
    """
    Test module with allreduce operation and redundant copy.
    
    This module creates a graph that contains:
    - A view operation
    - A linear transformation
    - An allreduce operation
    - A redundant copy_ operation (to be eliminated)
    - A bias addition
    """

    def __init__(self, in_dim=1024, out_dim=4096):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_dim, in_dim, dtype=torch.float16)
        )

        self.bias = nn.Parameter(
            torch.randn(out_dim, dtype=torch.float16)
        )

    def forward(self, x):
        """
        Forward pass with allreduce and redundant copy.
        
        Args:
            x: Input tensor with shape [batch_size, in_dim]
            
        Returns:
            tuple: (output_tensor, doubled_output_tensor)
        """
        # view
        x = x.view(-1, 1024)

        # linear
        out = F.linear(x, self.weight)
        doubled_output = out * 2
        # functional allreduce
        # pylint: disable=protected-access
        tensor = torch.ops._c10d_functional.all_reduce(
            out,
            "sum",
            "0"
        )

        # wait tensor
        # pylint: disable=protected-access
        wait_tensor = torch.ops._c10d_functional.wait_tensor(tensor)

        # inplace copy (this should be eliminated by copy_elimination pass)
        out.copy_(wait_tensor)

        # bias add
        out = out + self.bias

        return out, doubled_output


def run():
    """
    Execute the test with compiled model.
    
    This function:
    1. Creates model and moves to NPU device
    2. Creates random input tensor
    3. Compiles model with fx_backend
    4. Runs forward pass and prints output shapes
    """
    device = "npu"

    model = LinearAllReduceModule().to(device)

    x = torch.randn(
        47,
        1024,
        dtype=torch.float16,
        device=device
    )

    compiled_model = torch.compile(
        model,
        backend=fx_backend,
        fullgraph=True
    )

    y = compiled_model(x)

    print("output shape 1:", y[0].shape)
    print("output shape 2:", y[1].shape)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_eliminate_inplacecopy():
    """
    Feature: Test inplace copy eliminate.
    Description: Eliminate for inplace copy after wait_tensor.
    Expectation: The result is correct
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    dist.init_process_group(
        backend="hccl",
        world_size=1,
        rank=0
    )

    torch.npu.set_device(0)

    run()
