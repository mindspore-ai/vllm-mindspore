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
from mrt import jit
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dump", (True, False))
def test_jit(dump):
    """
    Feature: JIT compilation
    Description: Test JIT decorator functionality when dump_compiler is enabled/disabled
    Expectation: Function executes correctly and returns expected result in both dump modes
    """
    @jit(dump_compiler=dump)
    def run_check(x, y):
        print('hello world.\n')
        z = x * y
        z = z + x - y
        z = z / y
        return z

    assert run_check(12, 6) == 13
    print("The result is correct. 'mrt' module has been installed successfully.")