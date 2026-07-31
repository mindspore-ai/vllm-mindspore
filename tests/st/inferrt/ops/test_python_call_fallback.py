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
"""Tests for FX backend python_call fallback path.

The operators used below are wrapped by torch._dynamo.allow_in_graph so that
torch.compile treats each wrapper as a single FX node. Because the wrappers are
not registered as native InferRT ops nor as custom/torch ops, fx_backend falls
back to Op.python_call. The tests verify that the python_call argument
normalization (signature binding, default value filling, positional/keyword
alignment) works correctly for various input types and scenarios.
"""

# pylint: disable=protected-access
import pytest
import torch
import torch.nn.functional as F

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual

_allow_in_graph = torch._dynamo.allow_in_graph


@_allow_in_graph
def _multilabel_soft_margin_loss_call(x, y, weight, reduction):
    """Python-call wrapper for F.multilabel_soft_margin_loss."""
    return F.multilabel_soft_margin_loss(x, y, weight=weight, reduction=reduction)


@_allow_in_graph
def _normalize_call(x, p, dim, eps=1e-12):
    """Python-call wrapper for F.normalize."""
    return F.normalize(x, p=p, dim=dim, eps=eps)


@_allow_in_graph
def _poisson_nll_loss_call(x, y, log_input, full, eps, reduction):
    """Python-call wrapper for F.poisson_nll_loss."""
    return F.poisson_nll_loss(x, y, log_input=log_input, full=full, eps=eps, reduction=reduction)


@_allow_in_graph
def _mixed_type_python_call_op(
    x: torch.Tensor,
    scale: float,
    shift: int,
    flag: bool,
    label: str,
    extra_dims: list,
) -> torch.Tensor:
    """Custom Python op that exercises python_call with mixed scalar types."""
    y = x * scale + shift
    if flag:
        y = y * scale + shift
    else:
        y = y / scale - shift
    assert isinstance(label, str)
    for dim in extra_dims:
        y = y + dim
    return y


@_allow_in_graph
def _default_value_python_call_op(
    x: torch.Tensor,
    scale: float = 1.0,
    shift: int = 0,
    flag: bool = False,
    label: str = "default",
    extra_dims: list = None,
) -> torch.Tensor:
    """Custom Python op with default values to verify default filling."""
    if extra_dims is None:
        extra_dims = []
    y = x * scale + shift
    if flag:
        y = y * scale + shift
    assert isinstance(label, str)
    for dim in extra_dims:
        y = y + dim
    return y


@_allow_in_graph
def _cross_entropy_like_call(
    input_tensor: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Mimic the cross_entropy signature to verify positional/keyword alignment.

    The original cross_entropy bug occurred because a keyword argument like
    ``reduction`` was passed positionally, causing the string to land in the
    ``weight`` slot. This wrapper checks the types of the aligned arguments so
    that any misalignment raises the same kind of TypeError.
    """
    # pylint: disable=unused-argument

    if weight is not None and not isinstance(weight, torch.Tensor):
        raise TypeError(f"weight must be Tensor or None, got {type(weight).__name__}")
    if not isinstance(reduction, str):
        raise TypeError(f"reduction must be str, got {type(reduction).__name__}")
    return input_tensor


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("ignore_index", [-100, 0])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_python_call_cross_entropy_like_alignment(reduction, ignore_index, label_smoothing):
    """
    Feature: Test python_call positional/keyword alignment for cross_entropy-like signature
    Description: Pass only input/target and keyword args; verify they align correctly
                 so that the string reduction does not land in the weight slot.
    Expectation: Compiled output matches eager output without TypeError
    """
    torch.manual_seed(0)
    input_tensor = torch.randn((4, 8), dtype=torch.float32).npu()
    target = torch.randint(0, 8, (4,), dtype=torch.int64).npu()

    def cross_entropy_like_func(x, y):
        return _cross_entropy_like_call(
            x, y, reduction=reduction, ignore_index=ignore_index, label_smoothing=label_smoothing
        )

    eager_out = cross_entropy_like_func(input_tensor, target).detach().cpu()
    compiled_func = torch.compile(cross_entropy_like_func, backend=backend, fullgraph=True)
    compiled_out = compiled_func(input_tensor, target).detach().cpu()

    AssertRtolEqual(eager_out, compiled_out, prec=1e-3, prec16=2e-3)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_python_call_multilabel_soft_margin_loss(reduction):
    """
    Feature: Test multilabel_soft_margin_loss via python_call
    Description: Cover weighted reductions with float targets
    Expectation: Compiled output matches eager output
    """
    torch.manual_seed(0)
    input_tensor = torch.randn((4, 5), dtype=torch.float32).npu()
    target = torch.rand((4, 5), dtype=torch.float32).npu()
    weight = torch.rand((5,), dtype=torch.float32).npu()

    def loss_func(x, y, w):
        return _multilabel_soft_margin_loss_call(x, y, w, reduction)

    eager_out = loss_func(input_tensor, target, weight).detach().cpu()
    compiled_func = torch.compile(loss_func, backend=backend, fullgraph=True)
    compiled_out = compiled_func(input_tensor, target, weight).detach().cpu()

    AssertRtolEqual(eager_out, compiled_out, prec=1e-3, prec16=2e-3)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("p", [1.0, 2.0])
@pytest.mark.parametrize("dim", [0, 1])
def test_python_call_normalize(p, dim):
    """
    Feature: Test normalize via python_call
    Description: Cover L1/L2 normalization over different dims with float eps
    Expectation: Compiled output matches eager output
    """
    torch.manual_seed(0)
    input_tensor = torch.randn((3, 4), dtype=torch.float32).npu()

    def normalize_func(x):
        return _normalize_call(x, p, dim, 1e-8)

    eager_out = normalize_func(input_tensor).detach().cpu()
    compiled_func = torch.compile(normalize_func, backend=backend, fullgraph=True)
    compiled_out = compiled_func(input_tensor).detach().cpu()

    AssertRtolEqual(eager_out, compiled_out, prec=1e-3, prec16=2e-3)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("log_input", [True, False])
@pytest.mark.parametrize("full", [False, True])
def test_python_call_poisson_nll_loss(reduction, log_input, full):
    """
    Feature: Test poisson_nll_loss via python_call
    Description: Cover log_input/full flags and different reductions with float eps
    Expectation: Compiled output matches eager output
    """
    torch.manual_seed(0)
    if log_input:
        input_tensor = torch.log(torch.rand((4, 5), dtype=torch.float32) + 0.5).npu()
    else:
        input_tensor = torch.rand((4, 5), dtype=torch.float32).npu()
    target = torch.rand((4, 5), dtype=torch.float32).npu()

    def loss_func(x, y):
        return _poisson_nll_loss_call(x, y, log_input, full, 1e-8, reduction)

    eager_out = loss_func(input_tensor, target).detach().cpu()
    compiled_func = torch.compile(loss_func, backend=backend, fullgraph=True)
    compiled_out = compiled_func(input_tensor, target).detach().cpu()

    AssertRtolEqual(eager_out, compiled_out, prec=1e-3, prec16=2e-3)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize(
    "scale,shift,label,extra_dims",
    [
        (1.25, 2, "mixed_types", [1, 2, 3]),
        (0.5, -1, "python_call", [10, 20]),
    ],
)
def test_python_call_mixed_scalar_types(flag, scale, shift, label, extra_dims):
    """
    Feature: Test python_call with mixed scalar inputs
    Description: Verify tensor/float/int/bool/string/list are all passed correctly
    Expectation: Compiled output matches eager output
    """
    torch.manual_seed(0)
    x = torch.randn((4, 8), dtype=torch.float32).npu()

    def mixed_func(inp):
        return _mixed_type_python_call_op(inp, scale, shift, flag, label, extra_dims)

    eager_out = mixed_func(x).detach().cpu()
    compiled_func = torch.compile(mixed_func, backend=backend, fullgraph=True)
    compiled_out = compiled_func(x).detach().cpu()

    AssertRtolEqual(eager_out, compiled_out, prec=1e-3, prec16=2e-3)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((), {}),  # all default values
        ((2.0,), {}),  # one positional, rest default
        ((2.0, 1), {}),  # two positional, rest default
        ((2.0, 1, True), {"label": "partial_kw", "extra_dims": [1, 2]}),  # positional + keyword
        ((), {"scale": 2.0, "shift": 1, "flag": True, "label": "all_kw", "extra_dims": [3, 4]}),  # all keyword
    ],
)
def test_python_call_default_values(args, kwargs):
    """
    Feature: Test python_call with default values
    Description: Verify missing positional/keyword args are filled with defaults
    Expectation: Compiled output matches eager output
    """
    torch.manual_seed(0)
    x = torch.randn((4, 8), dtype=torch.float32).npu()

    def default_func(inp):
        return _default_value_python_call_op(inp, *args, **kwargs)

    eager_out = default_func(x).detach().cpu()
    compiled_func = torch.compile(default_func, backend=backend, fullgraph=True)
    compiled_out = compiled_func(x).detach().cpu()

    AssertRtolEqual(eager_out, compiled_out, prec=1e-3, prec16=2e-3)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("p", [1.0, 2.0])
@pytest.mark.parametrize("dim", [0, 1])
def test_python_call_normalize_default_eps(p, dim):
    """
    Feature: Test python_call with default eps
    Description: Verify omitted eps is filled with the wrapper default value
    Expectation: Compiled output matches eager output
    """
    torch.manual_seed(0)
    input_tensor = torch.randn((3, 4), dtype=torch.float32).npu()

    def normalize_func(x):
        return _normalize_call(x, p, dim)  # eps uses default

    eager_out = normalize_func(input_tensor).detach().cpu()
    compiled_func = torch.compile(normalize_func, backend=backend, fullgraph=True)
    compiled_out = compiled_func(input_tensor).detach().cpu()

    AssertRtolEqual(eager_out, compiled_out, prec=1e-3, prec16=2e-3)
