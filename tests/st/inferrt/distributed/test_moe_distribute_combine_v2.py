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

"""Tests for moe_distribute_combine_v2 distributed operation."""

import os

from tests.mark_utils import arg_mark

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPT_PATH = os.path.join(_CURRENT_DIR, "moe_distribute_combine_v2_runner.py")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_moe_distribute_combine_v2_basic():
    """
    Feature: Check moe_distribute_combine_v2 op launch
    Description: Check moe_distribute_combine_v2 op launch with basic configuration
    Expectation: The result is correct
    """

    cmd = f"torchrun --nproc_per_node=8 {_SCRIPT_PATH} test_moe_distribute_combine_v2_basic"
    return_code = os.system(cmd)
    assert return_code == 0


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_moe_distribute_combine_v2_with_tp():
    """
    Feature: Check moe_distribute_combine_v2 op launch with TP domain
    Description: Check moe_distribute_combine_v2 op launch with TP domain configuration
    Expectation: The result is correct
    """

    cmd = f"torchrun --nproc_per_node=8 {_SCRIPT_PATH} test_moe_distribute_combine_v2_with_tp"
    return_code = os.system(cmd)
    assert return_code == 0
