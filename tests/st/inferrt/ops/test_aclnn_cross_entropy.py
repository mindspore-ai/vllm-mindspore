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
"""Tests for cross_entropy via fx_backend using aclnnCrossEntropyLoss."""

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.exc import BackendCompilerFailed

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize(
    "reduction",
    ["mean", "sum", "none"],
)
def test_cross_entropy_fx_backend_basic(reduction):
    """
    Feature: Test cross_entropy with class-index targets via fx_backend
    Description: Cover mean/sum/none reductions for 2D logits and 1D targets
    Expectation: Result matches NPU eager output
    """
    torch.manual_seed(0)
    logits = torch.randn((4, 8), dtype=torch.float32).npu()
    target = torch.randint(0, 8, (4,), dtype=torch.int64).npu()

    def cross_entropy_func(x, y):
        return F.cross_entropy(x, y, reduction=reduction)

    eager_out = cross_entropy_func(logits, target).detach().cpu()
    compiled_func = torch.compile(cross_entropy_func, backend=backend, fullgraph=True)
    compiled_out = compiled_func(logits, target).detach().cpu()

    AssertRtolEqual(eager_out, compiled_out, prec=1e-3, prec16=2e-3)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize(
    "reduction",
    ["mean", "sum", "none"],
)
def test_cross_entropy_fx_backend_multidim(reduction):
    """
    Feature: Test cross_entropy with multi-dimensional logits via fx_backend
    Description: Cover mean/sum/none reductions for (N, C, D) logits and (N, D) targets
    Expectation: Result matches NPU eager output
    """
    torch.manual_seed(0)
    logits = torch.randn((2, 5, 3), dtype=torch.float32).npu()
    target = torch.randint(0, 5, (2, 3), dtype=torch.int64).npu()

    def cross_entropy_func(x, y):
        return F.cross_entropy(x, y, reduction=reduction)

    eager_out = cross_entropy_func(logits, target).detach().cpu()
    compiled_func = torch.compile(cross_entropy_func, backend=backend, fullgraph=True)
    compiled_out = compiled_func(logits, target).detach().cpu()

    AssertRtolEqual(eager_out, compiled_out, prec=1e-3, prec16=2e-3)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize(
    "reduction",
    ["mean", "sum"],
)
def test_cross_entropy_fx_backend_weighted(reduction):
    """
    Feature: Test cross_entropy with per-class weight via fx_backend
    Description: Cover weighted cross_entropy with mean/sum reductions
    Expectation: Result matches NPU eager output
    """
    torch.manual_seed(0)
    logits = torch.randn((4, 8), dtype=torch.float32).npu()
    target = torch.randint(0, 8, (4,), dtype=torch.int64).npu()
    weight = torch.rand((8,), dtype=torch.float32).npu()

    def cross_entropy_func(x, y, w):
        return F.cross_entropy(x, y, weight=w, reduction=reduction)

    eager_out = cross_entropy_func(logits, target, weight).detach().cpu()
    compiled_func = torch.compile(cross_entropy_func, backend=backend, fullgraph=True)
    compiled_out = compiled_func(logits, target, weight).detach().cpu()

    AssertRtolEqual(eager_out, compiled_out, prec=1e-3, prec16=2e-3)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize(
    "reduction",
    ["mean", "sum", "none"],
)
def test_cross_entropy_fx_backend_ignore_index(reduction):
    """
    Feature: Test cross_entropy with ignore_index via fx_backend
    Description: Cover ignore_index with mean/sum/none reductions
    Expectation: Result matches NPU eager output
    """
    torch.manual_seed(0)
    logits = torch.randn((4, 8), dtype=torch.float32).npu()
    target = torch.randint(0, 8, (4,), dtype=torch.int64).npu()
    # Make sure at least one target equals the ignored index.
    target[0] = 0

    def cross_entropy_func(x, y):
        return F.cross_entropy(x, y, ignore_index=0, reduction=reduction)

    eager_out = cross_entropy_func(logits, target).detach().cpu()
    compiled_func = torch.compile(cross_entropy_func, backend=backend, fullgraph=True)
    compiled_out = compiled_func(logits, target).detach().cpu()

    AssertRtolEqual(eager_out, compiled_out, prec=1e-3, prec16=2e-3)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize(
    "reduction",
    ["mean", "sum"],
)
def test_cross_entropy_fx_backend_label_smoothing(reduction):
    """
    Feature: Test cross_entropy with label_smoothing via fx_backend
    Description: Cover label_smoothing with mean/sum reductions
    Expectation: Result matches NPU eager output
    """
    torch.manual_seed(0)
    logits = torch.randn((4, 8), dtype=torch.float32).npu()
    target = torch.randint(0, 8, (4,), dtype=torch.int64).npu()

    def cross_entropy_func(x, y):
        return F.cross_entropy(x, y, label_smoothing=0.1, reduction=reduction)

    eager_out = cross_entropy_func(logits, target).detach().cpu()
    compiled_func = torch.compile(cross_entropy_func, backend=backend, fullgraph=True)
    compiled_out = compiled_func(logits, target).detach().cpu()

    AssertRtolEqual(eager_out, compiled_out, prec=1e-3, prec16=2e-3)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize(
    "reduction",
    ["mean", "sum"],
)
def test_cross_entropy_fx_backend_aten_hard_labels(reduction):
    """
    Feature: Test aten.cross_entropy_loss.default via fx_backend
    Description: Directly exercise the aten overload with class-index targets
    Expectation: Result matches NPU eager output
    """
    torch.manual_seed(0)
    logits = torch.randn((4, 8), dtype=torch.float16).npu()
    target = torch.randint(0, 8, (4,), dtype=torch.int64).npu()

    # aten.cross_entropy_loss.default schema expects int reduction (0/1/2);
    # convert the pytest string parameter to the corresponding enum.
    reduction_int = {"none": 0, "mean": 1, "sum": 2}[reduction]

    def cross_entropy_func(x, y, reduction=reduction_int):
        return torch.ops.aten.cross_entropy_loss.default(x, y, reduction=reduction)

    eager_out = F.cross_entropy(logits, target, reduction=reduction).detach().cpu()
    compiled_func = torch.compile(cross_entropy_func, backend=backend, fullgraph=True)
    compiled_out = compiled_func(logits, target).detach().cpu()

    AssertRtolEqual(eager_out, compiled_out, prec=1e-3, prec16=2e-3)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_cross_entropy_fx_backend_rejects_soft_labels():
    """
    Feature: Test soft labels are rejected via fx_backend
    Description: aclnnCrossEntropyLoss only supports class indices, not probability targets
    Expectation: Backend reports unsupported target type
    """
    torch.manual_seed(0)
    logits = torch.randn((4, 8), dtype=torch.float32).npu()
    target_probs = F.softmax(torch.randn((4, 8), dtype=torch.float32), dim=-1).npu()

    def cross_entropy_func(x, y):
        return F.cross_entropy(x, y)

    compiled_func = torch.compile(cross_entropy_func, backend=backend, fullgraph=True)

    with pytest.raises(BackendCompilerFailed, match="class-index targets"):
        compiled_func(logits, target_probs)
