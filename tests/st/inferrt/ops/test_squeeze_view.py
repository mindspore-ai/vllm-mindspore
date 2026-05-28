"""Tests for InferRT squeeze_view operator."""
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
@pytest.mark.parametrize("case", ["none", "dim", "dims", "method_dim", "non_unit_dim", "scalar", "scalar_dim"])
def test_squeeze_view_metadata(case):
    """
    Feature: Test squeeze_view
    Description: Cover squeeze None/dim/dims forms and scalar/non-unit-dim cases
    Expectation: The result and observable view metadata are consistent with eager mode
    """
    if case in ("scalar", "scalar_dim"):
        x = torch.rand((), dtype=torch.float32).npu()

        if case == "scalar":
            def func(input_x):
                return torch.squeeze(input_x)
        else:
            def func(input_x):
                return torch.squeeze(input_x, 0)

    else:
        x = torch.rand(2, 1, 3, 1, dtype=torch.float32).npu()

        if case == "none":
            def func(input_x):
                return torch.squeeze(input_x)
        elif case == "dim":
            def func(input_x):
                return torch.squeeze(input_x, 1)
        elif case == "dims":
            def func(input_x):
                return torch.squeeze(input_x, [1, 3])
        elif case == "method_dim":
            def func(input_x):
                return input_x.squeeze(-1)
        elif case == "non_unit_dim":
            def func(input_x):
                return torch.squeeze(input_x, 2)
        else:
            raise ValueError(f"unsupported squeeze case: {case}")

    compiled_func = torch.compile(func, backend=backend, fullgraph=True)
    output = compiled_func(x)
    expected = func(x)
    _assert_tensor_view_matches(expected, output)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("case", ["transpose_input", "permute_input"])
def test_squeeze_view_non_contiguous_input(case):
    """
    Feature: Test squeeze_view on non-contiguous inputs
    Description: Build non-contiguous view inputs before squeeze
    Expectation: The result and observable view metadata are consistent with eager mode
    """
    if case == "transpose_input":
        x = torch.rand(2, 3, 1, 4, dtype=torch.float32).transpose(0, 1).npu()

        def func(input_x):
            return input_x.squeeze(2)
    elif case == "permute_input":
        x = torch.rand(2, 3, 1, 4, dtype=torch.float32).permute(3, 1, 2, 0).npu()

        def func(input_x):
            return torch.squeeze(input_x, 2)
    else:
        raise ValueError(f"unsupported squeeze non-contiguous case: {case}")

    expected = func(x)
    assert not x.is_contiguous()
    compiled_func = torch.compile(func, backend=backend, fullgraph=True)
    output = compiled_func(x)
    _assert_tensor_view_matches(expected, output)
