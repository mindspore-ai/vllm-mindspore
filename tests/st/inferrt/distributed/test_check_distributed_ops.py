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
"""Tests for checking distributed ops launch with cache."""
import os

from tests.mark_utils import arg_mark

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPT_PATH = os.path.join(_CURRENT_DIR, "check_distributed_ops.py")

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_check_all_gather_op():
    """
    Feature: Check all_gather op launch
    Description: Check all_gather op launch with cache
    Expectation: The result is correct
    """
    cmd = f"torchrun --nproc_per_node=2 {_SCRIPT_PATH} test_all_gather"
    return_code = os.system(cmd)
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_check_reduce_scatter_op():
    """
    Feature: Check reduce_scatter op launch
    Description: Check reduce_scatter op launch with cache
    Expectation: The result is correct
    """
    cmd = f"torchrun --nproc_per_node=2 {_SCRIPT_PATH} test_reduce_scatter"
    return_code = os.system(cmd)
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_check_all_reduce_op():
    """
    Feature: Check all_reduce op launch
    Description: Check all_reduce op launch with cache
    Expectation: The result is correct
    """
    cmd = f"torchrun --nproc_per_node=2 {_SCRIPT_PATH} test_all_reduce"
    return_code = os.system(cmd)
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_check_all_to_all_single_op():
    """
    Feature: Check all_to_all_single op launch
    Description: Check all_to_all_single op launch with cache
    Expectation: The result is correct
    """
    cmd = f"torchrun --nproc_per_node=2 {_SCRIPT_PATH} test_all_to_all_single"
    return_code = os.system(cmd)
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_check_all_to_all_v_single_op():
    """
    Feature: Check all_to_all_v_single op launch
    Description: Check all_to_all_v_single op launch with cache
    Expectation: The result is correct
    """
    cmd = f"torchrun --nproc_per_node=2 {_SCRIPT_PATH} test_all_to_all_v_single"
    return_code = os.system(cmd)
    assert return_code == 0
