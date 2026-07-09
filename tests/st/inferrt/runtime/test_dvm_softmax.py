# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime regression test for DVM softmax kernels executed by InferRT."""

import math
import os

import pytest
import torch
from torch import nn

from ms_inferrt.torch.fx_backend import backend as fx_backend
from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual

os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "dvm"


class AttnSoftmaxRepro(nn.Module):
    """Small attention pattern that lets torch-npu DVM fuse the softmax region."""

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.scale = math.sqrt(head_dim)

    def forward(self, query, key, value):
        """Run causal attention with a softmax region that can be lowered to DVM."""
        seq = query.shape[2]
        kv = key.shape[2]
        heads = query.shape[1]
        mask = torch.tril(torch.ones(seq, kv, device=query.device))
        query = query.view(heads, seq, query.shape[3])
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, value)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.skip(reason="DVM softmax runtime case is kept as a manual regression repro.")
def test_dvm_softmax_runtime():
    """
    Feature: DVM softmax kernel execution through InferRT
    Description: Compile an attention softmax pattern with torch-npu DVM fusion and execute it via InferRT backend
    Expectation: The compiled output matches eager output
    """
    device = "npu"
    heads = 8
    head_dim = 128
    seq = 1
    kv = 1
    torch.manual_seed(0)

    query = torch.randn(1, heads, seq, head_dim, device=device)
    key = torch.randn(1, heads, kv, head_dim, device=device)
    value = torch.randn(1, heads, kv, head_dim, device=device)
    model = AttnSoftmaxRepro(head_dim).to(device)
    tdcv4 = pytest.importorskip("torch_dispatch_capture.v4")

    with torch.no_grad():
        expected = model(query, key, value)

    with torch.no_grad(), tdcv4.enable_device_with_fusion(device, fx_backend):
        compiled_model = torch.compile(model, backend="inductor", dynamic=True)
        output = compiled_model(query, key, value)
        torch.npu.synchronize()

    AssertRtolEqual(output, expected)
