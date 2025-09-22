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

import torch
from mrt.torch import backend
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_kernel_group_launch(monkeypatch):
    """
    Feature: Test kernel group launch(parallel launch)
    Description: Test 2 threads parallel launch and total kernel group 8
    Expectation: The result is correct
    """

    monkeypatch.setenv("MRT_KERNEL_LAUNCH_THREAD_NUM", "2")
    monkeypatch.setenv("MRT_KERNEL_LAUNCH_GROUP_NUM", "8")

    def foo(x, y):
        out = torch.mul(x, y)
        for i in range(20):
            out = torch.mul(x, out)
            out = torch.mul(y, out)
        return torch.mul(x, out)

    opt_foo = torch.compile(foo, backend=backend)

    # first run by static shape
    x = torch.randn(2, 2).npu()
    y = torch.randn(2, 2).npu()
    _ = opt_foo(x, y)

    z = torch.randn(3, 3).npu()
    q = torch.randn(3, 3).npu()
    # Second run by dynamic shape
    _ = opt_foo(z, q)
    out = None
    # Run by parallel launch mode.
    for i in range(10):
        out = opt_foo(z, q)

    origin_out = foo(z, q)

    assert torch.equal(origin_out, out), f"\norigin_out={origin_out}\nout={out}"
    print("The result is correct")
