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

"""
Tests for OpTorchCall ref/view/inplace/normal handling.

These ops are deliberately chosen to NOT be in the frontend _OP_MAP, so they fall through to the
custom_call -> OpTorchCall path. That exercises the schema-alias based ref detection
(OpTorchCall::ComputeRefPairsFromSchema) and the ownership hand-off in OpTorchCall::ToFxrtTensor:

- view     : torch.ops.aten.ravel   (writer-free alias, refPairs_ = {(0,0)})
- inplace  : x.relu_()              (write alias,       refPairs_ = {(0,0)})
- normal   : torch.ops.aten.ldexp   (no alias,          refPairs_ = {} -> independent output)

For the view/inplace cases the runtime aliases the output to the input's Storage, so taking ownership
of the data a second time would double-free it; these tests assert that does not happen (the process
must exit cleanly) and that the observable semantics hold.
"""
import torch
from tests.common import HasTorchNpu
from tests.mark_utils import arg_mark

if HasTorchNpu():
    import torch_npu    # pylint: disable=unused-import

# backend must load after torch_npu: ms_inferrt.torch.__init__ imports torch_npu first to set LD_LIBRARY_PATH.
# pylint: disable=wrong-import-position
from ms_inferrt.torch import backend


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_torch_call_view_ravel():
    """
    Feature: OpTorchCall ref detection for a view operator.
    Description: torch.ops.aten.ravel carries a writer-free alias on its schema
                 (Tensor(a self) -> Tensor(a)); ComputeRefPairsFromSchema must record refPairs_={(0,0)},
                 so the output shares the input's Storage and ToFxrtTensor must not reclaim ownership.
    Expectation: The result is correct and the graph exits cleanly (no double-free).
    """
    def op(x):
        return torch.ops.aten.ravel(x)

    compiled = torch.compile(op, backend=backend)
    x = torch.randn(4, 4).npu()
    result = compiled(x)

    assert tuple(result.shape) == (16,), f"unexpected ravel shape: {tuple(result.shape)}"
    assert torch.allclose(result.cpu(), x.cpu().ravel()), (
        f"ravel result mismatch\nexpected={x.cpu().ravel()}\nresult={result.cpu()}"
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_torch_call_inplace_relu_():
    """
    Feature: OpTorchCall ref detection for an inplace operator.
    Description: x.relu_() carries a writer alias on its schema (Tensor(a! self) -> Tensor(a!));
                 ComputeRefPairsFromSchema records refPairs_={(0,0)} (write alias is not filtered),
                 so the output aliases the input Storage. The inplace semantics must survive the run:
                 the caller's tensor is rewritten, and the returned tensor carries that result.
    Expectation: The input tensor is rewritten in place and the result is correct; process exits
                 cleanly (no double-free).
    """
    def op(x):
        return x.relu_()

    compiled = torch.compile(op, backend=backend)
    x = torch.randn(4, 4).npu()
    x_copy = x.cpu().clone()
    result = compiled(x)

    assert torch.allclose(result.cpu(), x_copy.relu()), (
        f"relu_ result mismatch\nexpected={x_copy.relu()}\nresult={result.cpu()}"
    )
    assert torch.allclose(x.cpu(), x_copy.relu()), (
        "relu_ must rewrite the input tensor in place"
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_torch_call_view_detach():
    """
    Feature: OpTorchCall ref detection for a second view operator.
    Description: torch.ops.aten.detach also carries a writer-free alias (Tensor(a self) -> Tensor(a)),
                 so it must be recognised as a ref/view and exit cleanly without a double-free.
    Expectation: The result is correct and the graph exits cleanly.
    """
    def op(x):
        return torch.ops.aten.detach(x)

    compiled = torch.compile(op, backend=backend)
    x = torch.randn(4, 4).npu()
    result = compiled(x)

    assert tuple(result.shape) == tuple(x.shape)
    assert torch.allclose(result.cpu(), x.cpu()), (
        f"detach result mismatch\nexpected={x.cpu()}\nresult={result.cpu()}"
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_torch_call_normal_ldexp():
    """
    Feature: OpTorchCall normal (non-ref) path.
    Description: torch.ops.aten.ldexp has no alias annotation, so refPairs_ stays empty and the output
                 is freshly allocated (ToFxrtTensor takes ownership as usual).
    Expectation: The result is correct and independent of the inputs.
    """
    def op(x, y):
        return torch.ops.aten.ldexp(x, y)

    compiled = torch.compile(op, backend=backend)
    x = torch.randn(2, 16).npu()
    y = torch.zeros(2, 16, dtype=torch.float32).npu()
    result = compiled(x, y)

    expected = torch.ops.aten.ldexp(x, y)
    assert torch.allclose(result.cpu(), expected.cpu()), (
        f"ldexp result mismatch\nexpected={expected.cpu()}\nresult={result.cpu()}"
    )
    assert result.data_ptr() != x.data_ptr(), "ldexp output must be an independent tensor"
