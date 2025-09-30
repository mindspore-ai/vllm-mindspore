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
import os

import pytest
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_check_all_gather_op(pipeline, monkeypatch):
    """
    Feature: Check all_gather op launch
    Description: Check all_gather op launch with cache
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    command = "torchrun --nproc_per_node=2 tests/st/inferrt/distributed/check_distributed_ops.py test_all_gather"
    return_code = os.system(command)
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_check_reduce_scatter_op(pipeline, monkeypatch):
    """
    Feature: Check reduce_scatter op launch
    Description: Check reduce_scatter op launch with cache
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    command = "torchrun --nproc_per_node=2 tests/st/inferrt/distributed/check_distributed_ops.py test_reduce_scatter"
    return_code = os.system(command)
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_check_all_reduce_op(pipeline, monkeypatch):
    """
    Feature: Check all_reduce op launch
    Description: Check all_reduce op launch with cache
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    command = "torchrun --nproc_per_node=2 tests/st/inferrt/distributed/check_distributed_ops.py test_all_reduce"
    return_code = os.system(command)
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
def test_check_all_to_all_op(pipeline, monkeypatch):
    """
    Feature: Check all_to_all op launch
    Description: Check all_to_all op launch with cache
    Expectation: The result is correct
    """
    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")
    command = "torchrun --nproc_per_node=2 tests/st/inferrt/distributed/check_distributed_ops.py test_all_to_all"
    return_code = os.system(command)
    assert return_code == 0
