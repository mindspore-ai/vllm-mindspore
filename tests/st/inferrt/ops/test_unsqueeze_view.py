# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Tests for InferRT unsqueeze_view operator."""
import pytest
import torch
import torch._dynamo.config as dynamo_config

from ms_inferrt.torch.fx_backend import backend
from ms_inferrt.torch.fx_mlir_backend import backend as mlir_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual

dynamo_config.cache_size_limit = 64


def _assert_tensor_view_matches(expected, actual):
    assert tuple(actual.shape) == tuple(expected.shape)
    assert tuple(actual.stride()) == tuple(expected.stride())
    AssertRtolEqual(expected.detach().cpu(), actual.detach().cpu())


# pylint: disable=redefined-builtin
def op_func(input, dim):
    """Reference implementation of unsqueeze."""
    return input.unsqueeze(dim)


def get_op_func_compiled():
    """Get compiled unsqueeze function for the MLIR backend."""

    def custom_op_func(x, dim):
        return x.unsqueeze(dim)

    return torch.compile(custom_op_func, backend=mlir_backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("infos", [([2, 3, 4, 9], 2)])
@pytest.mark.parametrize("dtypes", [torch.float16, torch.bfloat16, torch.float32])
def test_unsqueeze(infos, dtypes):
    """
    Feature: Test unsqueeze
    Description: Preserve existing MLIR backend unsqueeze coverage
    Expectation: The result is correct
    """
    cpu_input0 = torch.rand(infos[0], dtype=dtypes)
    npu_input0 = cpu_input0.npu()
    dim0 = infos[1]
    cpu_output = op_func(cpu_input0, dim0)
    op_func_compiled = get_op_func_compiled()
    npu_output = op_func_compiled(npu_input0, dim0)
    npu_output_cpu = npu_output.cpu()
    AssertRtolEqual(cpu_output, npu_output_cpu)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("case", ["front", "middle", "tail", "scalar"])
def test_unsqueeze_view_metadata(case):
    """
    Feature: Test unsqueeze_view
    Description: Cover positive/negative dim and scalar unsqueeze cases
    Expectation: The result and observable view metadata are consistent with eager mode
    """
    if case == "scalar":
        x = torch.rand((), dtype=torch.float32).npu()

        def func(input_x):
            return input_x.unsqueeze(0)

    else:
        x = torch.rand(2, 3, 4, dtype=torch.float32).npu()

        if case == "front":
            def func(input_x):
                return torch.unsqueeze(input_x, 0)
        elif case == "middle":
            def func(input_x):
                return input_x.unsqueeze(2)
        elif case == "tail":
            def func(input_x):
                return torch.unsqueeze(input_x, -1)
        else:
            raise ValueError(f"unsupported unsqueeze case: {case}")

    compiled_func = torch.compile(func, backend=backend, fullgraph=True)
    output = compiled_func(x)
    expected = func(x)
    _assert_tensor_view_matches(expected, output)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("case", ["transpose_input", "slice_input"])
def test_unsqueeze_view_non_contiguous_input(case):
    """
    Feature: Test unsqueeze_view on non-contiguous inputs
    Description: Build non-contiguous view inputs before unsqueeze
    Expectation: The result and observable view metadata are consistent with eager mode
    """
    if case == "transpose_input":
        x = torch.rand(2, 3, 4, dtype=torch.float32).transpose(0, 1).npu()

        def func(input_x):
            return input_x.unsqueeze(1)
    elif case == "slice_input":
        x = torch.rand(2, 3, 4, dtype=torch.float32).npu()[:, :, ::2]

        def func(input_x):
            return torch.unsqueeze(input_x, -1)
    else:
        raise ValueError(f"unsupported unsqueeze non-contiguous case: {case}")

    expected = func(x)
    assert not x.is_contiguous()
    compiled_func = torch.compile(func, backend=backend, fullgraph=True)
    output = compiled_func(x)
    _assert_tensor_view_matches(expected, output)
