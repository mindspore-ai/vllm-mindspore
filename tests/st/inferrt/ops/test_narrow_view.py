"""Tests for InferRT narrow_view operator."""
import pytest
import torch
import torch._dynamo.config as dynamo_config

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual

dynamo_config.cache_size_limit = 64


def _assert_tensor_view_matches(expected, actual):
    assert tuple(actual.shape) == tuple(expected.shape)
    assert tuple(actual.stride()) == tuple(expected.stride())
    AssertRtolEqual(expected.detach().cpu(), actual.detach().cpu())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("case", ["torch_positive", "method_negative", "sliced_input"])
def test_narrow_view_metadata(case):
    """
    Feature: Test narrow_view
    Description: Cover torch/method forms, negative start, and non-zero input storage offset
    Expectation: The result and observable view metadata are consistent with eager mode
    """
    x = torch.rand(4, 5, 6, dtype=torch.float32).npu()

    if case == "torch_positive":
        def func(input_x):
            return torch.narrow(input_x, 1, 1, 3)
    elif case == "method_negative":
        def func(input_x):
            return input_x.narrow(-1, -4, 2)
    elif case == "sliced_input":
        def func(input_x):
            return input_x[1:].narrow(1, 0, 2)
    else:
        raise ValueError(f"unsupported narrow case: {case}")

    compiled_func = torch.compile(func, backend=backend, fullgraph=True)
    output = compiled_func(x)
    expected = func(x)
    _assert_tensor_view_matches(expected, output)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("case", ["transpose_input", "permute_input"])
def test_narrow_view_non_contiguous_input(case):
    """
    Feature: Test narrow_view on non-contiguous inputs
    Description: Build non-contiguous view inputs before narrow
    Expectation: The result and observable view metadata are consistent with eager mode
    """
    if case == "transpose_input":
        x = torch.rand(4, 5, 6, dtype=torch.float32).transpose(0, 1).npu()

        def func(input_x):
            return input_x.narrow(0, 1, 3)
    elif case == "permute_input":
        x = torch.rand(4, 5, 6, dtype=torch.float32).permute(2, 0, 1).npu()

        def func(input_x):
            return torch.narrow(input_x, 2, -4, 2)
    else:
        raise ValueError(f"unsupported narrow non-contiguous case: {case}")

    expected = func(x)
    assert not x.is_contiguous()
    compiled_func = torch.compile(func, backend=backend, fullgraph=True)
    output = compiled_func(x)
    _assert_tensor_view_matches(expected, output)
