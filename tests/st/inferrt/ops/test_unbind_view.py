"""Tests for InferRT unbind_view operator."""
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
@pytest.mark.parametrize("case", ["torch_dim", "method_negative_dim"])
def test_unbind_view_metadata(case):
    """
    Feature: Test unbind_view
    Description: Cover torch/method unbind forms with tuple view outputs
    Expectation: The results and observable view metadata are consistent with eager mode
    """
    x = torch.rand(2, 3, 4, dtype=torch.float32).npu()

    if case == "torch_dim":
        def func(input_x):
            return torch.unbind(input_x, 1)
    elif case == "method_negative_dim":
        def func(input_x):
            return input_x.unbind(-1)
    else:
        raise ValueError(f"unsupported unbind case: {case}")

    compiled_func = torch.compile(func, backend=backend, fullgraph=True)
    outputs = compiled_func(x)
    expected_outputs = func(x)
    assert len(outputs) == len(expected_outputs)
    for expected, output in zip(expected_outputs, outputs):
        _assert_tensor_view_matches(expected, output)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_unbind_view_consumed_by_getitem():
    """
    Feature: Test unbind_view tuple consumption
    Description: Verify tuple_getitem consumers can read unbind_view outputs
    Expectation: The result is consistent with eager mode
    """

    def func(input_x):
        return torch.unbind(input_x, 1)[2] + 1.0

    x = torch.rand(2, 3, 4, dtype=torch.float32).npu()
    compiled_func = torch.compile(func, backend=backend, fullgraph=True)
    output = compiled_func(x)
    expected = func(x)
    AssertRtolEqual(expected.detach().cpu(), output.detach().cpu())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("case", ["transpose_input", "permute_input"])
def test_unbind_view_non_contiguous_input(case):
    """
    Feature: Test unbind_view on non-contiguous inputs
    Description: Build non-contiguous view inputs before unbind
    Expectation: The results and observable view metadata are consistent with eager mode
    """
    if case == "transpose_input":
        x = torch.rand(2, 3, 4, dtype=torch.float32).transpose(0, 1).npu()

        def func(input_x):
            return torch.unbind(input_x, 0)
    elif case == "permute_input":
        x = torch.rand(2, 3, 4, dtype=torch.float32).permute(2, 0, 1).npu()

        def func(input_x):
            return input_x.unbind(-1)
    else:
        raise ValueError(f"unsupported unbind non-contiguous case: {case}")

    expected_outputs = func(x)
    assert not x.is_contiguous()
    compiled_func = torch.compile(func, backend=backend, fullgraph=True)
    outputs = compiled_func(x)
    assert len(outputs) == len(expected_outputs)
    for expected, output in zip(expected_outputs, outputs):
        _assert_tensor_view_matches(expected, output)
