# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/engine/arg_utils.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2025 The vLLM team.
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
"""Adaption for arguments utils"""

import sys

from vllm.config import ModelConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

logger = init_logger(__name__)


def _set_default_args(self, usage_context: UsageContext,
                      model_config: ModelConfig) -> None:
    """Set Default Arguments for V1 Engine."""

    # V1 always uses chunked prefills and prefix caching
    # for non-pooling tasks.
    # For pooling tasks the default is False
    if model_config.runner_type != "pooling":
        # vllm-mindspore: Need to support the `False`
        if self.enable_chunked_prefill is None:
            self.enable_chunked_prefill = True

        # TODO: When prefix caching supports prompt embeds inputs, this
        # check can be removed.
        if (self.enable_prompt_embeds
                and self.enable_prefix_caching is not False):
            logger.warning(
                "--enable-prompt-embeds and --enable-prefix-caching "
                "are not supported together in V1. Prefix caching has "
                "been disabled.")
            self.enable_prefix_caching = False

        if self.enable_prefix_caching is None:
            self.enable_prefix_caching = True
    else:

        pooling_type = model_config.pooler_config.pooling_type
        is_causal = getattr(model_config.hf_config, "is_causal", True)
        incremental_prefill_supported = (pooling_type is not None
                                         and pooling_type.lower() == "last"
                                         and is_causal)

        action = "Enabling" if \
            incremental_prefill_supported else "Disabling"

        if self.enable_chunked_prefill is None:
            self.enable_chunked_prefill = incremental_prefill_supported
            logger.info("(%s) chunked prefill by default", action)
        if self.enable_prefix_caching is None:
            self.enable_prefix_caching = incremental_prefill_supported
            logger.info("(%s) prefix caching by default", action)

    # V1 should use the new scheduler by default.
    # Swap it only if this arg is set to the original V0 default
    if self.scheduler_cls == EngineArgs.scheduler_cls:
        # vllm-mindspore: Use EnhancedScheduler in place of Scheduler
        self.scheduler_cls = (
            "vllm_mindspore.v1.core.sched.scheduler.EnhancedScheduler")

    # When no user override, set the default values based on the usage
    # context.
    # Use different default values for different hardware.

    # vllm-mindspore: Get device memory will initialize device runtime, which
    # will be inherited by the child process in fork mode, resulting in
    # setting device failure for latter ASCEND_RT_VISIBLE_DEVICES modification.
    # So skip it.
    from vllm.usage.usage_lib import UsageContext

    # TODO(woosuk): Tune the default values for other hardware.
    default_max_num_batched_tokens = {
        UsageContext.LLM_CLASS: 8192,
        UsageContext.OPENAI_API_SERVER: 2048,
    }
    default_max_num_seqs = {
        UsageContext.LLM_CLASS: 256,
        UsageContext.OPENAI_API_SERVER: 256,
    }

    use_context_value = usage_context.value if usage_context else None
    if (self.max_num_batched_tokens is None
            and usage_context in default_max_num_batched_tokens):
        if not self.enable_chunked_prefill:
            self.max_num_batched_tokens = model_config.max_model_len
        else:
            self.max_num_batched_tokens = \
                default_max_num_batched_tokens[usage_context]
        logger.debug(
            "Setting max_num_batched_tokens to %d for %s usage context.",
            self.max_num_batched_tokens, use_context_value)

    if (self.max_num_seqs is None and usage_context in default_max_num_seqs):
        self.max_num_seqs = min(default_max_num_seqs[usage_context],
                                self.max_num_batched_tokens or sys.maxsize)

        logger.debug("Setting max_num_seqs to %d for %s usage context.",
                     self.max_num_seqs, use_context_value)
