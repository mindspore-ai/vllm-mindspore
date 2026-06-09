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
Tests for aten to torch schema replacement in op selection.
"""
import torch

from ms_inferrt.torch.fx_backend import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def mul_scalar(x):
    """Mul with scalar to exercise schema replacement."""
    return torch.ops.aten.mul.Tensor(x, 2.0)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_aten_mul_tensor_scalar_schema_replacement():
    """
    Feature: Schema replacement for aten ops during op selection
    Description: aten.mul.Tensor with scalar input triggers schema replacement to torch.mul
    Expectation: Graph compiles and runs correctly after schema replacement
    """
    torch.npu.set_device(0)

    compiled_func = torch.compile(mul_scalar, backend=fx_backend)

    x = torch.randn(4, 4, dtype=torch.float32).npu()
    expected = mul_scalar(x)
    result = compiled_func(x)

    AssertRtolEqual(expected.detach().cpu().numpy(), result.detach().cpu().numpy())
