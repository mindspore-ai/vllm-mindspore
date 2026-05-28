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
"""Tests for InferRT view outputs consumed by non-view ref-like ops."""

from typing import Callable, Tuple

import pytest
import torch
import torch._dynamo.config as dynamo_config
import torch_npu

from ms_inferrt.torch.fx_backend import backend

from tests.mark_utils import arg_mark  # pylint: disable=import-error


dynamo_config.cache_size_limit = 64
_NZ_FORMAT = 29

TensorArgs = Tuple[torch.Tensor, ...]
TensorOutput = torch.Tensor | Tuple[torch.Tensor, ...]
CaseFunc = Callable[..., TensorOutput]
CaseArgsBuilder = Callable[[], TensorArgs]


def _to_tuple(output: TensorOutput) -> Tuple[torch.Tensor, ...]:
    if isinstance(output, tuple):
        return output
    return (output,)


def _clone_args(args: TensorArgs) -> TensorArgs:
    return tuple(arg.clone() for arg in args)


def _logical_metadata(tensor: torch.Tensor):
    return {
        "shape": tuple(tensor.shape),
        "stride": tuple(tensor.stride()),
        "is_contiguous": tensor.is_contiguous(),
    }


def _assert_outputs_match(eager_outputs, compiled_outputs):
    assert len(eager_outputs) == len(compiled_outputs)
    for eager, compiled in zip(eager_outputs, compiled_outputs):
        assert _logical_metadata(compiled) == _logical_metadata(eager)
        torch.testing.assert_close(
            compiled.detach().cpu(),
            eager.detach().cpu(),
            rtol=1e-3,
            atol=1e-3,
        )


def _make_view_fill_args() -> TensorArgs:
    return (torch.randn((4, 4), dtype=torch.float32).npu(),)


def _view_fill_scalar(x):
    view = x.view(4, 4)
    updated = view.fill_(2.0)
    return updated


def _inplace_add(x, y):
    updated = x.add_(y)
    return updated


def _index_put(x, index, value):
    updated = x.index_put_((index,), value)
    return updated


def _make_reshape_add_scalar_args() -> TensorArgs:
    return (torch.randn((2, 3, 4), dtype=torch.float32).npu(),)


def _reshape_add_scalar(x):
    view = torch.reshape(x, (6, 4))
    updated = view.add_(1.0)
    return updated


def _make_permute_add_args() -> TensorArgs:
    x = torch.randn((4, 4), dtype=torch.float32).npu()
    y = torch.randn((4, 4), dtype=torch.float32).npu()
    return x, y


def _permute_add(x, y):
    view = torch.movedim(x, 0, 1)
    updated = view.add_(y)
    return updated


def _make_slice_add_scalar_args() -> TensorArgs:
    return (torch.randn((5, 4), dtype=torch.float32).npu(),)


def _slice_add_scalar(x):
    view = x[1:4]
    updated = view.add_(1.25)
    return updated


def _make_select_fill_tensor_args() -> TensorArgs:
    x = torch.randn((4, 3), dtype=torch.float32).npu()
    value = torch.tensor(3.0, dtype=torch.float32).npu()
    return x, value


def _select_fill_tensor(x, value):
    view = torch.select(x, 0, 2)
    updated = view.fill_(value)
    return updated


def _make_narrow_index_copy_args() -> TensorArgs:
    x = torch.randn((6, 3), dtype=torch.float32).npu()
    index = torch.tensor([0, 2], dtype=torch.int64).npu()
    source = torch.randn((2, 3), dtype=torch.float32).npu()
    return x, index, source


def _narrow_index_copy(x, index, source):
    view = torch.narrow(x, 0, 1, 4)
    updated = view.index_copy_(0, index, source)
    return updated


def _make_split_with_size_index_put_args() -> TensorArgs:
    x = torch.randn((6, 3), dtype=torch.float32).npu()
    index = torch.tensor([0, 2], dtype=torch.int64).npu()
    value = torch.randn((2, 3), dtype=torch.float32).npu()
    return x, index, value


def _split_with_size_index_put(x, index, value):
    _, view = torch.split(x, [3, 3], dim=0)
    updated = view.index_put_((index,), value)
    return updated


def _make_split_tensor_add_scalar_args() -> TensorArgs:
    return (torch.randn((6, 3), dtype=torch.float32).npu(),)


def _split_tensor_add_scalar(x):
    _, view, _ = torch.split(x, 2, dim=0)
    updated = view.add_(0.5)
    return updated


def _make_chunk_fill_scalar_args() -> TensorArgs:
    return (torch.randn((6, 3), dtype=torch.float32).npu(),)


def _chunk_fill_scalar(x):
    _, view, _ = torch.chunk(x, 3, dim=0)
    updated = view.fill_(5.0)
    return updated


def _make_squeeze_masked_fill_scalar_args() -> TensorArgs:
    x = torch.randn((2, 1, 3), dtype=torch.float32).npu()
    mask = torch.zeros((2, 3), dtype=torch.bool).npu()
    mask[0, 1] = True
    mask[1, 2] = True
    return x, mask


def _squeeze_masked_fill_scalar(x, mask):
    view = x.squeeze(1)
    updated = view.masked_fill_(mask, -3.0)
    return updated


def _make_unsqueeze_masked_fill_tensor_args() -> TensorArgs:
    x = torch.randn((2, 3), dtype=torch.float32).npu()
    mask = torch.zeros((1, 2, 3), dtype=torch.bool).npu()
    mask[0, 0, 1] = True
    mask[0, 1, 2] = True
    value = torch.tensor(-2.0, dtype=torch.float32).npu()
    return x, mask, value


def _unsqueeze_masked_fill_tensor(x, mask, value):
    view = x.unsqueeze(0)
    updated = view.masked_fill_(mask, value)
    return updated


def _make_unbind_fill_scalar_args() -> TensorArgs:
    return (torch.randn((3, 4), dtype=torch.float32).npu(),)


def _unbind_fill_scalar(x):
    _, view, _ = torch.unbind(x, dim=0)
    updated = view.fill_(4.0)
    return updated


def _make_flatten_scatter_nd_update_args() -> TensorArgs:
    x = torch.zeros((2, 3, 4), dtype=torch.float32).npu()
    indices = torch.tensor([[0, 1], [5, 2]], dtype=torch.int64).npu()
    updates = torch.tensor([7.0, 9.0], dtype=torch.float32).npu()
    return x, indices, updates


def _flatten_scatter_nd_update(x, indices, updates):
    view = torch.flatten(x, 0, 1)
    updated = torch.ops.npu.npu_scatter_nd_update_(view, indices, updates)
    return updated


def _make_rotary_args() -> TensorArgs:
    out = torch.randn((4, 1280), dtype=torch.float16).npu()
    cos = torch.randn((4, 128), dtype=torch.float16).npu()
    sin = torch.randn((4, 128), dtype=torch.float16).npu()
    return out, cos, sin


def _rotary_query_after_split_view(out, cos, sin):
    """Return rotary query output after splitting a packed tensor view."""
    q, k, _ = torch.split(out, [1024, 128, 128], dim=-1)
    q = q.view(out.shape[0], 8, 128)
    k = k.view(out.shape[0], 1, 128)
    query_rot, _ = torch.ops.npu.npu_apply_rotary_pos_emb(
        q[None, ...],
        k[None, ...],
        cos[None, :, None, :],
        sin[None, :, None, :],
        "BSND",
    )
    return query_rot.squeeze(0).view(-1, 8, 128)


def _rotary_key_after_split_view(out, cos, sin):
    """Return rotary key output after splitting a packed tensor view."""
    q, k, _ = torch.split(out, [1024, 128, 128], dim=-1)
    q = q.view(out.shape[0], 8, 128)
    k = k.view(out.shape[0], 1, 128)
    _, key_rot = torch.ops.npu.npu_apply_rotary_pos_emb(
        q[None, ...],
        k[None, ...],
        cos[None, :, None, :],
        sin[None, :, None, :],
        "BSND",
    )
    return key_rot.squeeze(0).view(-1, 1, 128)


def _run_case(func: CaseFunc, make_args: CaseArgsBuilder):
    base_args = make_args()
    eager_outputs = _to_tuple(func(*_clone_args(base_args)))

    torch.compiler.reset()
    compiled_func = torch.compile(func, backend=backend, fullgraph=True)
    compiled_outputs = _to_tuple(compiled_func(*_clone_args(base_args)))
    _assert_outputs_match(eager_outputs, compiled_outputs)


def _run_special_format_rejected(func: CaseFunc, args: TensorArgs):
    torch.compiler.reset()
    compiled_func = torch.compile(func, backend=backend, fullgraph=True)
    with pytest.raises(RuntimeError, match=r"special-format ref metadata sync"):
        compiled_func(*args)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_view_shape_equal_fill_scalar_metadata_match_eager():
    """
    Feature: InferRT view plus non-view ref-like operators
    Description: Verify shape-equal view metadata is preserved by fill_
    Expectation: Compiled outputs match eager mode
    """
    _run_case(_view_fill_scalar, _make_view_fill_args)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_reshape_view_add_scalar_metadata_match_eager():
    """
    Feature: InferRT view plus non-view ref-like operators
    Description: Verify reshape lowering to view preserves metadata for add_
    Expectation: Compiled outputs match eager mode
    """
    _run_case(_reshape_add_scalar, _make_reshape_add_scalar_args)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_permute_view_add_metadata_match_eager():
    """
    Feature: InferRT view plus non-view ref-like operators
    Description: Verify permute_view strides are visible to add_
    Expectation: Compiled outputs match eager mode
    """
    _run_case(_permute_add, _make_permute_add_args)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_slice_view_add_scalar_metadata_match_eager():
    """
    Feature: InferRT view plus non-view ref-like operators
    Description: Verify slice_view storage offset is visible to add_
    Expectation: Compiled outputs match eager mode
    """
    _run_case(_slice_add_scalar, _make_slice_add_scalar_args)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_select_view_fill_tensor_metadata_match_eager():
    """
    Feature: InferRT view plus non-view ref-like operators
    Description: Verify select_view metadata is visible to fill_
    Expectation: Compiled outputs match eager mode
    """
    _run_case(_select_fill_tensor, _make_select_fill_tensor_args)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_narrow_view_index_copy_metadata_match_eager():
    """
    Feature: InferRT view plus non-view ref-like operators
    Description: Verify narrow_view storage offset is visible to index_copy_
    Expectation: Compiled outputs match eager mode
    """
    _run_case(_narrow_index_copy, _make_narrow_index_copy_args)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_split_with_size_view_index_put_metadata_match_eager():
    """
    Feature: InferRT view plus non-view ref-like operators
    Description: Verify split_with_size_view offset is visible to index_put_
    Expectation: Compiled outputs match eager mode
    """
    _run_case(_split_with_size_index_put, _make_split_with_size_index_put_args)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_split_tensor_view_add_scalar_metadata_match_eager():
    """
    Feature: InferRT view plus non-view ref-like operators
    Description: Verify split_tensor_view offset is visible to add_
    Expectation: Compiled outputs match eager mode
    """
    _run_case(_split_tensor_add_scalar, _make_split_tensor_add_scalar_args)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_chunk_to_split_with_size_view_fill_scalar_metadata_match_eager():
    """
    Feature: InferRT view plus non-view ref-like operators
    Description: Verify chunk lowering to split_with_size_view preserves metadata for fill_
    Expectation: Compiled outputs match eager mode
    """
    _run_case(_chunk_fill_scalar, _make_chunk_fill_scalar_args)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_squeeze_view_masked_fill_scalar_metadata_match_eager():
    """
    Feature: InferRT view plus non-view ref-like operators
    Description: Verify squeeze_view metadata is visible to masked_fill_
    Expectation: Compiled outputs match eager mode
    """
    _run_case(_squeeze_masked_fill_scalar, _make_squeeze_masked_fill_scalar_args)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_unsqueeze_view_masked_fill_tensor_metadata_match_eager():
    """
    Feature: InferRT view plus non-view ref-like operators
    Description: Verify unsqueeze_view metadata is visible to masked_fill_
    Expectation: Compiled outputs match eager mode
    """
    _run_case(_unsqueeze_masked_fill_tensor, _make_unsqueeze_masked_fill_tensor_args)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_unbind_view_fill_scalar_metadata_match_eager():
    """
    Feature: InferRT view plus non-view ref-like operators
    Description: Verify unbind_view offset is visible to fill_
    Expectation: Compiled outputs match eager mode
    """
    _run_case(_unbind_fill_scalar, _make_unbind_fill_scalar_args)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_flatten_view_scatter_nd_update_metadata_match_eager():
    """
    Feature: InferRT view plus non-view ref-like operators
    Description: Verify flatten view metadata is visible to scatter_nd_update_
    Expectation: Compiled outputs match eager mode
    """
    _run_case(_flatten_scatter_nd_update, _make_flatten_scatter_nd_update_args)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_rotary_query_after_split_view_metadata_match_eager():
    """
    Feature: InferRT view plus non-view ref-like operators
    Description: Verify apply_rotary_pos_emb keeps query metadata after view chain
    Expectation: Compiled outputs match eager mode
    """
    _run_case(_rotary_query_after_split_view, _make_rotary_args)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_rotary_key_after_split_view_metadata_match_eager():
    """
    Feature: InferRT view plus non-view ref-like operators
    Description: Verify apply_rotary_pos_emb keeps key metadata after view chain
    Expectation: Compiled outputs match eager mode
    """
    _run_case(_rotary_key_after_split_view, _make_rotary_args)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_inplace_add_special_format_metadata_sync_rejected():
    """
    Feature: InferRT ref-like metadata sync
    Description: Reject special-format tensors before generic ref metadata sync drops layout
    Expectation: Compiled execution raises an explicit special-format ref metadata error
    """
    x = torch.randn((16, 32), dtype=torch.float16).npu()
    x = torch_npu.npu_format_cast(x.contiguous(), _NZ_FORMAT)  # pylint: disable=no-member
    y = torch.randn((16, 32), dtype=torch.float16).npu()
    _run_special_format_rejected(_inplace_add, (x, y))


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_index_put_special_format_metadata_sync_rejected():
    """
    Feature: InferRT ref-like metadata sync
    Description: Reject special-format tensors before generic ref metadata sync drops layout
    Expectation: Compiled execution raises an explicit special-format ref metadata error
    """
    x = torch.randn((16, 32), dtype=torch.float16).npu()
    x = torch_npu.npu_format_cast(x.contiguous(), _NZ_FORMAT)  # pylint: disable=no-member
    index = torch.tensor([0, 2], dtype=torch.int64).npu()
    value = torch.randn((2, 32), dtype=torch.float16).npu()
    _run_special_format_rejected(_index_put, (x, index, value))
