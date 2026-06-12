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
"""Empty tensor coverage for InferRT view operators."""
import math
from typing import Callable, Tuple

import pytest
import torch
import torch._dynamo.config as dynamo_config
import torch_npu  # pylint: disable=unused-import

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark


dynamo_config.cache_size_limit = 128

TensorOutput = torch.Tensor | Tuple[torch.Tensor, ...]
ViewFunc = Callable[[torch.Tensor], TensorOutput]
InputBuilder = Callable[[], torch.Tensor]


def _empty(shape, dtype=torch.float16):
    return torch.empty(shape, dtype=dtype)


def _arange(shape, dtype=torch.float16):
    return torch.arange(math.prod(shape), dtype=dtype).reshape(shape)


def _assert_tensor_view_matches(expected: torch.Tensor, actual: torch.Tensor):
    assert tuple(actual.shape) == tuple(expected.shape)
    assert tuple(actual.stride()) == tuple(expected.stride())
    if expected.numel() == 0:
        assert actual.storage_offset() == expected.storage_offset()
    assert actual.is_contiguous() == expected.is_contiguous()
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu())


def _assert_output_matches(expected: TensorOutput, actual: TensorOutput):
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(actual) == len(expected)
        for expected_tensor, actual_tensor in zip(expected, actual):
            _assert_tensor_view_matches(expected_tensor, actual_tensor)
        return
    assert isinstance(actual, torch.Tensor)
    _assert_tensor_view_matches(expected, actual)


def _transpose_empty_front(x):
    return torch.transpose(x, 0, 1)


def _permute_empty_middle(x):
    return x.permute(2, 0, 1)


def _movedim_empty_middle(x):
    return torch.movedim(x, 2, 0)


def _flatten_empty_leading_dims(x):
    return x.flatten(0, 1)


def _flatten_empty_all_dims(x):
    return torch.flatten(x, 0, -1)


def _unsqueeze_empty_front(x):
    return x.unsqueeze(0)


def _unsqueeze_empty_tail(x):
    return torch.unsqueeze(x, -1)


def _squeeze_empty_unit_dim(x):
    return x.squeeze(0)


def _squeeze_empty_non_unit_dim(x):
    return torch.squeeze(x, 1)


def _narrow_zero_length_nonzero_offset(x):
    return x.narrow(0, 1, 0)


def _slice_zero_length_nonzero_offset_step(x):
    return torch.ops.aten.slice.Tensor(x, 0, 1, 1, 2)


def _slice_empty_input_with_step(x):
    return torch.ops.aten.slice.Tensor(x, 0, 0, 0, 2)


def _select_empty_result_from_zero_storage(x):
    return torch.select(x, 0, 2)


def _select_method_empty_result_from_zero_storage(x):
    return x.select(0, 2)


def _unbind_empty_outputs(x):
    return torch.unbind(x, 0)


def _split_tensor_empty_input(x):
    return x.split(1, dim=0)


def _split_with_size_zero_middle(x):
    return torch.split(x, [1, 0, 3], dim=0)


def _chunk_empty_input(x):
    return torch.chunk(x, 3, dim=0)


_SINGLE_OUTPUT_CASES = [
    ("transpose_empty_front", lambda: _empty((0, 5)), _transpose_empty_front),
    ("permute_empty_middle", lambda: _empty((2, 0, 5)), _permute_empty_middle),
    ("movedim_empty_middle", lambda: _empty((2, 0, 5)), _movedim_empty_middle),
    ("flatten_empty_leading_dims", lambda: _empty((0, 2, 3)), _flatten_empty_leading_dims),
    ("flatten_empty_all_dims", lambda: _empty((2, 0, 3)), _flatten_empty_all_dims),
    ("unsqueeze_empty_front", lambda: _empty((0, 5)), _unsqueeze_empty_front),
    ("unsqueeze_empty_tail", lambda: _empty((0, 5)), _unsqueeze_empty_tail),
    ("squeeze_empty_unit_dim", lambda: _empty((1, 0, 5)), _squeeze_empty_unit_dim),
    ("squeeze_empty_non_unit_dim", lambda: _empty((1, 0, 5)), _squeeze_empty_non_unit_dim),
    ("narrow_zero_length_nonzero_offset", lambda: _arange((4, 5)), _narrow_zero_length_nonzero_offset),
    ("slice_zero_length_nonzero_offset_step", lambda: _arange((4, 5)), _slice_zero_length_nonzero_offset_step),
    ("slice_empty_input_with_step", lambda: _empty((0, 5)), _slice_empty_input_with_step),
    ("select_empty_result_from_zero_storage", lambda: _empty((4, 0, 5)), _select_empty_result_from_zero_storage),
    (
        "select_method_empty_result_from_zero_storage",
        lambda: _empty((4, 0, 5)),
        _select_method_empty_result_from_zero_storage,
    ),
]

_TUPLE_OUTPUT_CASES = [
    ("unbind_empty_outputs", lambda: _empty((3, 0, 5)), _unbind_empty_outputs),
    ("split_tensor_empty_input", lambda: _empty((2, 0, 5)), _split_tensor_empty_input),
    ("split_with_size_zero_middle", lambda: _arange((4, 5)), _split_with_size_zero_middle),
    ("chunk_empty_input", lambda: _empty((0, 5)), _chunk_empty_input),
]

_SPLIT_EMPTY_COMPILE_BACKENDS = [
    ("inductor", None),
    ("inferrt", backend),
]


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "case_name,input_builder,func",
    _SINGLE_OUTPUT_CASES,
    ids=[case[0] for case in _SINGLE_OUTPUT_CASES],
)
def test_empty_tensor_single_output_view_metadata(case_name: str, input_builder: InputBuilder, func: ViewFunc):
    """
    Feature: InferRT view operators on empty tensors
    Description: Verify single-output view ops preserve eager empty tensor metadata and values
    Expectation: Shape, stride, storage_offset, contiguity and values are consistent with torch_npu and CPU eager
    """
    del case_name
    cpu_x = input_builder()
    npu_x = cpu_x.npu()
    cpu_expected = func(cpu_x)
    npu_expected = func(npu_x)
    compiled_func = torch.compile(func, backend=backend, fullgraph=True)
    actual = compiled_func(npu_x)
    _assert_output_matches(npu_expected, actual)
    _assert_output_matches(cpu_expected, actual)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "case_name,input_builder,func",
    _TUPLE_OUTPUT_CASES,
    ids=[case[0] for case in _TUPLE_OUTPUT_CASES],
)
def test_empty_tensor_tuple_output_view_metadata(case_name: str, input_builder: InputBuilder, func: ViewFunc):
    """
    Feature: InferRT tuple-output view operators on empty tensors
    Description: Verify unbind/split/chunk outputs preserve eager empty tensor metadata and values
    Expectation: Each tuple element matches torch_npu and CPU eager shape, stride, storage_offset, contiguity and value
    """
    del case_name
    cpu_x = input_builder()
    npu_x = cpu_x.npu()
    cpu_expected = func(cpu_x)
    npu_expected = func(npu_x)
    compiled_func = torch.compile(func, backend=backend, fullgraph=True)
    actual = compiled_func(npu_x)
    _assert_output_matches(npu_expected, actual)
    _assert_output_matches(cpu_expected, actual)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("backend_name,compile_backend", _SPLIT_EMPTY_COMPILE_BACKENDS, ids=["inductor", "inferrt"])
def test_torch_compile_split_empty_dim_raises_before_backend(backend_name, compile_backend):
    """
    Feature: torch.compile empty split failure coverage
    Description: Verify split(int) on an empty split dimension fails during Dynamo FakeTensor tracing
    Expectation: The known PyTorch decomposition error is raised before backend execution
    """
    del backend_name

    def func(x):
        return torch.split(x, 2, dim=0)

    x = torch.empty(0, 6, dtype=torch.float16).npu()
    if compile_backend is None:
        compiled_func = torch.compile(func, fullgraph=True)
    else:
        compiled_func = torch.compile(func, backend=compile_backend, fullgraph=True)

    # torch.compile evaluates aten.split.Tensor with FakeTensor before invoking the selected backend.
    # For dim_size == 0 and split_size > 0, PyTorch's decomposition builds an empty split_sizes list
    # and then writes split_sizes[-1], so both default inductor and InferRT backends see the same error.
    with pytest.raises(RuntimeError, match="list assignment index out of range"):
        compiled_func(x)
    # Keep the stream drained after the expected host-side exception. This avoids leaving InferRT's
    # default async launch thread active if a future code path reaches backend execution before failing.
    torch.npu.synchronize()
