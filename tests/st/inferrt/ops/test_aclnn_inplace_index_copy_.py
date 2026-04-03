"""Tests for aclnn inplace index_copy_."""

import pytest
import torch

from ms_inferrt.torch.fx_backend import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def _run_and_compare(backend, dtype, dim, self_shape, index, source_shape):
    """Compare eager vs torch.compile index_copy_ outputs on NPU."""
    # Reference: eager execution on NPU.
    self_eager = torch.randn(*self_shape, dtype=dtype).npu()
    source = torch.randn(*source_shape, dtype=dtype).npu()
    index = index.npu()

    eager_out = self_eager.clone()
    eager_out.index_copy_(dim, index, source)

    def op_func(x, idx, src):
        return x.index_copy_(dim, idx, src)

    compiled_op = torch.compile(op_func, backend=backend)
    self_graph = self_eager.clone()
    graph_out = compiled_op(self_graph, index, source)

    prec = 1e-3 if dtype == torch.float16 else 1e-4
    AssertRtolEqual(eager_out, graph_out, prec=prec, prec16=1e-3)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("backend", (fx_backend,))
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
@pytest.mark.parametrize(
    "dim,self_shape,index_list,source_shape",
    [
        # dim = 0
        (0, (4, 5), [3, 1, 0], (3, 5)),
        # dim = -1
        (-1, (2, 4, 3), [2, 0], (2, 4, 2)),
    ],
)
def test_aclnn_inplace_index_copy_(backend, dtype, dim, self_shape, index_list, source_shape):
    """
    Feature: aclnn inplace index_copy_ via torch.compile fx_backend.
    Description: float16/float32 tensors on NPU; dim 0 or -1 with varied self/index/source shapes.
    Expectation: compiled index_copy_ output matches eager index_copy_ within rtol tolerance.
    """
    index = torch.tensor(index_list, dtype=torch.int64)
    _run_and_compare(
        backend=backend,
        dtype=dtype,
        dim=dim,
        self_shape=self_shape,
        index=index,
        source_shape=source_shape,
    )
