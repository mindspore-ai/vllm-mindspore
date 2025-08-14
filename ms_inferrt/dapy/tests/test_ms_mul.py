# Copyright 2025 Zhang Qinghua
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

import numpy as np
import mindspore as ms
from mindspore import ops
from dapy import dag


@dag(dump_compiler=True)
def mul_func(x, y):
    return ops.mul(x, y)

def test_mul():
    x = ms.Tensor(np.array([1, 2, 4]).astype(np.float32))
    y = ms.Tensor(np.array([2, 4, 3]).astype(np.float32))
    return mul_func(x, y)


output = test_mul()
expect_output = np.array([2., 8., 12.]).astype(np.float32)
np.testing.assert_array_equal(output.asnumpy(), expect_output)
print("The result is correct. Interaction with MindSpore successfully.")