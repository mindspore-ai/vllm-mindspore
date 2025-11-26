# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Only used for adapter to ms-pipeline, do not add case here"""
import pytest


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_empty_level0_910b_case():
    assert True


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_empty_level1_910b_case():
    assert True


@pytest.mark.level0
@pytest.mark.platform_ascend310p
@pytest.mark.env_single
def test_empty_level0_310p_case():
    assert True


@pytest.mark.level1
@pytest.mark.platform_ascend310p
@pytest.mark.env_single
def test_empty_level1_310p_case():
    assert True
