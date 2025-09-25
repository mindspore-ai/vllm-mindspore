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

"""test_fx_mlir_backend"""

import torch
import numpy as np
from mrt.torch.fx_mlir_backend import backend


def foo(x, y):
    return torch.matmul(x, y)


opt_foo = torch.compile(foo, backend=backend)


def run(shape1, shape2):
    x_np = np.random.randn(*shape1).astype(np.float32)
    y_np = np.random.randn(*shape2).astype(np.float32)
    expect = np.matmul(x_np, y_np)
    out = opt_foo(torch.tensor(x_np), torch.tensor(y_np))
    assert np.allclose(out, expect, 1e-3, 1e-3), f"\nout={out}\nexpect={expect}"


run((2, 2), (2, 4))

print("The result is correct. 'mrt' backend has been installed successfully.")
