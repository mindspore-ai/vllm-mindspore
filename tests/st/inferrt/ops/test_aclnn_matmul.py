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

import pytest
import numpy as np
import torch
from torch_npu.testing.common_utils import create_common_tensor

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual
from mrt.torch import backend


PrescsionTableFP16 = [
    [2, 1e2, 0.005], [2, 1e3, 0.005], [2, 1e4, 0.005], [2, 1e5, 0.005], [2, 1e6, 0.005],
    [10, 1e2, 0.005], [10, 1e3, 0.01], [10, 1e4, 0.02], [10, 1e5, 0.0305], [10, 1e6, 0.04],
    [50, 1e2, 0.03], [50, 1e3, 0.03], [50, 1e4, 0.03], [50, 1e5, 0.03], [50, 1e6, 0.04],
    [100, 1e2, 0.03], [100, 1e3, 0.03], [100, 1e4, 0.03], [100, 1e5, 0.03], [100, 1e6, 0.04],
    [1000, 1e2, 0.03], [1000, 1e3, 0.04], [1000, 1e4, 0.04], [1000, 1e5, 0.04], [1000, 1e6, 0.04],
    [10000, 1e2, 0.04], [10000, 1e3, 0.04], [10000, 1e4, 0.04], [10000, 1e5, 0.04], [10000, 1e6, 0.04],
]

def getFp16Precsion(D_range, K_range):
    """Get the precision for fp16"""
    prec16 = 1e-3
    for elm in PrescsionTableFP16:
        if elm[0] == D_range and elm[1] == K_range:
            return elm[2]
    return prec16

def AssertRtolEqualMatmul(x, y):
    """
    Assert that two tensors are equal within a relative tolerance for matmul
    Args:
        x: The first tensor.
        y: The second tensor.
    """
    D = np.amax(np.maximum(np.abs(x), np.abs(y))) if (x.size and y.size) else 1
    D_range = 10000
    D_range = 10000 if (D > 1000) else D_range
    D_range = 1000 if (D <= 1000) else D_range
    D_range = 100 if (D <= 100) else D_range
    D_range = 50 if (D <= 50) else D_range
    D_range = 2 if (D <= 2) else D_range

    Kx = max(x.shape) if x.shape else 1
    Ky = max(y.shape) if y.shape else 1
    K = max(Kx, Ky)
    K_range = 1e6
    K_range = 1e6 if (K > 1e5) else K_range
    K_range = 1e5 if (K <= 1e5) else K_range
    K_range = 1e4 if (K <= 1e4) else K_range
    K_range = 1e3 if (K <= 1e3) else K_range
    K_range = 1e2 if (K <= 1e2) else K_range

    prec16 = 1e-3
    if x.dtype == np.float16 or x.dtype == np.float32:
        prec16 = getFp16Precsion(D_range, K_range)

    AssertRtolEqual(x, y, prec16, prec16)

def op_func(mat1, mat2):
    """op function for matmul"""
    return torch.matmul(mat1, mat2)

def matmul_forward(shape_format, op_func_compiled):
    """
    matmul forward function
    Args:
        shape_format: The shape format of the input.
        op_func_compiled: The compiled op function.
    """
    for item in shape_format:
        mat1_cpu, mat1_npu = create_common_tensor(item[0], -10, 10)
        if mat1_cpu.dtype == torch.float16:
            mat1_cpu = mat1_cpu.to(torch.float32)
        mat2_cpu, mat2_npu = create_common_tensor(item[1], -10, 10)
        if mat2_cpu.dtype == torch.float16:
            mat2_cpu = mat2_cpu.to(torch.float32)
        cpu_output = op_func(mat1_cpu, mat2_cpu).detach().numpy()
        npu_output = op_func_compiled(mat1_npu, mat2_npu).detach().cpu().numpy()

        AssertRtolEqualMatmul(cpu_output.astype(npu_output.dtype), npu_output)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_matmul_shape_format_fp16_case1(pipeline, monkeypatch):
    """
    Feature: Test aclnn matmul
    Description: Test aclnn matmul with 1-D inputs
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    shape_format = [
        # mat1 1dim, mat2 1dim
        [[np.float16, 2, [5]], [np.float16, 2, [5]]],
        [[np.float16, 2, [16]], [np.float16, 2, [16]]],
    ]
    op_func_compiled = torch.compile(op_func, backend=backend)
    matmul_forward(shape_format, op_func_compiled)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_matmul_shape_format_fp16_case2(pipeline, monkeypatch):
    """
    Feature: Test aclnn matmul
    Description: Test aclnn matmul with 1-D mat1 and 2-D mat2
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    shape_format = [
        # mat1 1dim, mat2 2dim
        [[np.float16, 2, [5]], [np.float16, 2, [5, 6]]],
        [[np.float16, 2, [5]], [np.float16, 2, [5, 5]]],
    ]
    op_func_compiled = torch.compile(op_func, backend=backend)
    matmul_forward(shape_format, op_func_compiled)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_matmul_shape_format_fp16_case3(pipeline, monkeypatch):
    """
    Feature: Test aclnn matmul
    Description: Test aclnn matmul with 2-D inputs
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    shape_format = [
        # mat1 2dim, mat2 2dim
        [[np.float16, 2, [5, 7]], [np.float16, 2, [7, 10]]],
        [[np.float16, 2, [5, 10]], [np.float16, 2, [10, 20]]],
    ]
    op_func_compiled = torch.compile(op_func, backend=backend)
    matmul_forward(shape_format, op_func_compiled)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_matmul_shape_format_fp16_case4(pipeline, monkeypatch):
    """
    Feature: Test aclnn matmul
    Description: Test aclnn matmul with >2-D mat1 and 1-D mat2
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    shape_format = [
        # mat1 >2dim, mat2 1dim
        [[np.float16, 2, [4, 5, 10]], [np.float16, 2, [10]]],
        [[np.float16, 2, [5, 10, 20, 30]], [np.float16, 2, [30]]],
        [[np.float16, 2, [20, 30, 40, 50, 60]], [np.float16, 2, [60]]],
        [[np.float16, 2, [2, 3, 4, 5, 6, 8]], [np.float16, 2, [8]]],
    ]
    op_func_compiled = torch.compile(op_func, backend=backend)
    matmul_forward(shape_format, op_func_compiled)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_matmul_shape_format_fp16_case5(pipeline, monkeypatch):
    """
    Feature: Test aclnn matmul
    Description: Test aclnn matmul with >2-D mat1 and 2-D mat2
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    shape_format = [
        # mat1 >2dim, mat2 2dim
        [[np.float16, 2, [5, 7, 10]], [np.float16, 2, [10, 16]]],
        [[np.float16, 2, [5, 10, 20, 30]], [np.float16, 2, [30, 25]]],
        [[np.float16, 2, [2, 5, 7, 8, 9, 10]], [np.float16, 2, [10, 16]]],
    ]
    op_func_compiled = torch.compile(op_func, backend=backend)
    matmul_forward(shape_format, op_func_compiled)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_matmul_shape_format_fp16_case6(pipeline, monkeypatch):
    """
    Feature: Test aclnn matmul
    Description: Test aclnn matmul with 1-D mat1 and >2-D mat2
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    shape_format = [
        # mat1 1dim, mat2 >2dim
        [[np.float16, 2, [3]], [np.float16, 2, [2, 3, 2]]],
        [[np.float16, 2, [20]], [np.float16, 2, [5, 10, 20, 30]]],
    ]
    op_func_compiled = torch.compile(op_func, backend=backend)
    matmul_forward(shape_format, op_func_compiled)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_matmul_shape_format_fp16_case7(pipeline, monkeypatch):
    """
    Feature: Test aclnn matmul
    Description: Test aclnn matmul with 2-D mat1 and >2-D mat2
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    shape_format = [
        # mat1 2dim, mat2 >2dim
        [[np.float16, 2, [2, 3]], [np.float16, 2, [2, 3, 2]]],
        [[np.float16, 2, [44, 20]], [np.float16, 2, [5, 10, 20, 30]]],
        [[np.float16, 2, [75, 50]], [np.float16, 2, [2, 3, 40, 50, 60]]],
    ]
    op_func_compiled = torch.compile(op_func, backend=backend)
    matmul_forward(shape_format, op_func_compiled)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_matmul_shape_format_fp16_case8(pipeline, monkeypatch):
    """
    Feature: Test aclnn matmul
    Description: Test aclnn matmul with >2-D inputs
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    shape_format = [
        [[np.float16, 2, [5, 7, 10]], [np.float16, 2, [5, 10, 15]]],
        [[np.float16, 2, [68, 75, 16]], [np.float16, 2, [68, 16, 43]]],
    ]
    op_func_compiled = torch.compile(op_func, backend=backend)
    matmul_forward(shape_format, op_func_compiled)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_matmul_allow_hf32(pipeline, monkeypatch):
    """
    Feature: Test aclnn matmul with HF32 enabled
    Description: Test aclnn matmul with allow_hf32=True
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    torch.npu.matmul.allow_hf32 = True
    shape_format = [
        # mat1 1dim, mat2 1dim
        [[np.float16, 2, [5]], [np.float16, 2, [5]]],
        [[np.float16, 2, [16]], [np.float16, 2, [16]]],
    ]
    op_func_compiled = torch.compile(op_func, backend=backend)
    matmul_forward(shape_format, op_func_compiled)
    torch.npu.matmul.allow_hf32 = False


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_matmul_opapi(pipeline, monkeypatch):
    """
    Feature: Test aclnn matmul with allow_hf32=True
    Description: Test aclnn matmul with complex shapes using allow_hf32=True
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    torch.npu.matmul.allow_hf32 = True
    shape_format = [
        [[np.float16, 2, [1, 1, 10, 2, 16, 16]], [np.float16, 2, [1, 10, 1, 16, 16]]],
        [[np.float16, 2, [1, 11, 10, 10, 16, 5]], [np.float16, 2, [1, 10, 1, 5, 16]]],
        [[np.float16, 2, [400, 11, 10, 10, 16, 5]], [np.float16, 2, [1, 10, 1, 5, 16]]],
    ]
    op_func_compiled = torch.compile(op_func, backend=backend)
    matmul_forward(shape_format, op_func_compiled)
    torch.npu.matmul.allow_hf32 = False


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_matmul_shape_diff_input_types(pipeline, monkeypatch):
    """
    Feature: Test aclnn matmul with allow_hf32=True
    Description: Test aclnn matmul with mixed float16 and float32 inputs
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    torch.npu.matmul.allow_hf32 = True
    shape_format = [
        [[np.float16, 2, [1, 7, 10]], [np.float32, 2, [5, 10, 15]]],
        [[np.float32, 2, [68, 75, 16]], [np.float16, 2, [16, 43]]],
    ]
    op_func_compiled = torch.compile(op_func, backend=backend)
    matmul_forward(shape_format, op_func_compiled)
    torch.npu.matmul.allow_hf32 = False
