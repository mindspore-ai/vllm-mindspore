#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
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
# ============================================================================
"""init logger for vllm-mindspore."""

from logging.config import dictConfig
import vllm.envs as envs
from vllm.logger import DEFAULT_LOGGING_CONFIG, init_logger

VLLM_CONFIGURE_LOGGING = envs.VLLM_CONFIGURE_LOGGING
VLLM_LOGGING_CONFIG_PATH = envs.VLLM_LOGGING_CONFIG_PATH
VLLM_LOGGING_LEVEL = envs.VLLM_LOGGING_LEVEL
VLLM_LOGGING_PREFIX = envs.VLLM_LOGGING_PREFIX

_DATE_FORMAT = "%m-%d %H:%M:%S"
_MS_FORMAT = (f"{VLLM_LOGGING_PREFIX}%(levelname)s %(asctime)s "
           "vllm-mindspore[%(filename)s:%(lineno)d] %(message)s")

_MS_FORMATTERS = {
    "vllm_mindspore": {
        "class": "vllm.logging_utils.NewLineFormatter",
        "datefmt": _DATE_FORMAT,
        "format": _MS_FORMAT,
    }
}

_MS_HANDLERS = {
    "vllm_mindspore": {
        "class": "logging.StreamHandler",
        "formatter": "vllm_mindspore",
        "level": VLLM_LOGGING_LEVEL,
        "stream": "ext://sys.stdout",
    }
}

_MS_LOGGERS = {
    "vllm_mindspore": {
        "handlers": ["vllm_mindspore"],
        "level": "DEBUG",
        "propagate": False,
    }
}

def _update_configure_vllm_root_logger() -> None:
    if VLLM_CONFIGURE_LOGGING and not VLLM_LOGGING_CONFIG_PATH:
        logging_config = DEFAULT_LOGGING_CONFIG
        logging_config["formatters"].update(_MS_FORMATTERS)
        logging_config["handlers"].update(_MS_HANDLERS)
        logging_config["loggers"].update(_MS_LOGGERS)

        dictConfig(logging_config)

_update_configure_vllm_root_logger()

logger = init_logger(__name__)
logger.info("The config of vllm-mindspore logger has been updated successfully.")
