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

import vllm_mindspore.logger
from vllm.logger import init_logger

import vllm_mindspore.envs as env
from vllm_mindspore.plugins_register.utils import init_env, init_context


logger = init_logger(__name__)


def register_models():
    if not env.ENABLE_MS_ADAPTER:
        return

    init_env()
    init_context()

    logger.info("Register MindSpore models via vllm.general_plugins.")

    from vllm_mindspore.plugins_register.register_models import (
        _register_ms_models)
    _register_ms_models()
