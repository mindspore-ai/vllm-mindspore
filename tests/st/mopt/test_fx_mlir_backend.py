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


def foo(x):
    """Reshape a 2x3 tensor to 3x2"""
    return x.reshape(3, 2)


opt_foo = torch.compile(foo, backend=backend)


def run(input_shape, output_shape):
    """Test reshape operation"""
    x_np = np.random.randn(*input_shape).astype(np.float32)
    x_tensor = torch.tensor(x_np)
    
    # Expected result using numpy
    expect = x_np.reshape(output_shape)
    
    # Result from compiled backend
    out = opt_foo(x_tensor)
    
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected shape: {output_shape}")
    
    assert out.shape == output_shape, f"Shape mismatch: {out.shape} vs {output_shape}"
    assert np.allclose(out.numpy(), expect, 1e-5, 1e-5), f"\nout={out}\nexpect={expect}"


# Test reshape from 2x3 to 3x2
run((2, 3), (3, 2))

print("The result is correct. 'mrt' backend has been installed successfully.")
