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
"""Adaption for arguments utils."""

import threading
from typing import get_args

import torch
import vllm.envs as envs
from vllm.config import (GuidedDecodingBackendV1, LoadFormat, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.engine.arg_utils import (EngineArgs, _raise_or_fallback,
                                   _warn_or_fallback)
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

logger = init_logger(__name__)


def _is_v1_supported_oracle(self, model_config: ModelConfig) -> bool:
    """Oracle for whether to use V0 or V1 Engine by default."""

    #############################################################
    # Unsupported Feature Flags on V1.

    if self.load_format == LoadFormat.SHARDED_STATE.value:
        _raise_or_fallback(feature_name=f"--load_format {self.load_format}",
                           recommend_to_remove=False)
        return False

    if (self.logits_processor_pattern != EngineArgs.logits_processor_pattern):
        _raise_or_fallback(feature_name="--logits-processor-pattern",
                           recommend_to_remove=False)
        return False

    if self.preemption_mode != SchedulerConfig.preemption_mode:
        _raise_or_fallback(feature_name="--preemption-mode",
                           recommend_to_remove=True)
        return False

    if (self.disable_async_output_proc
            != EngineArgs.disable_async_output_proc):
        _raise_or_fallback(feature_name="--disable-async-output-proc",
                           recommend_to_remove=True)
        return False

    if self.scheduling_policy != SchedulerConfig.policy:
        _raise_or_fallback(feature_name="--scheduling-policy",
                           recommend_to_remove=False)
        return False

    if self.num_scheduler_steps != SchedulerConfig.num_scheduler_steps:
        _raise_or_fallback(feature_name="--num-scheduler-steps",
                           recommend_to_remove=True)
        return False

    if self.scheduler_delay_factor != SchedulerConfig.delay_factor:
        _raise_or_fallback(feature_name="--scheduler-delay-factor",
                           recommend_to_remove=True)
        return False

    if self.guided_decoding_backend not in get_args(GuidedDecodingBackendV1):
        _raise_or_fallback(
            feature_name=
            f"--guided-decoding-backend={self.guided_decoding_backend}",
            recommend_to_remove=False)
        return False

    # Need at least Ampere for now (FA support required).
    # Skip this check if we are running on a non-GPU platform,
    # or if the device capability is not available
    # (e.g. in a Ray actor without GPUs).
    from vllm.platforms import current_platform
    if (current_platform.is_cuda()
            and current_platform.get_device_capability()
            and current_platform.get_device_capability().major < 8):
        _raise_or_fallback(feature_name="Compute Capability < 8.0",
                           recommend_to_remove=False)
        return False

    # No Fp8 KV cache so far.
    if self.kv_cache_dtype != "auto":
        fp8_attention = self.kv_cache_dtype.startswith("fp8")
        will_use_fa = (current_platform.is_cuda()
                       and not envs.is_set("VLLM_ATTENTION_BACKEND")
                       ) or envs.VLLM_ATTENTION_BACKEND == "FLASH_ATTN_VLLM_V1"
        supported = False
        if current_platform.is_rocm():
            supported = True
        elif fp8_attention and will_use_fa:
            from vllm.attention.utils.fa_utils import flash_attn_supports_fp8
            supported = flash_attn_supports_fp8()
        if not supported:
            _raise_or_fallback(feature_name="--kv-cache-dtype",
                               recommend_to_remove=False)
            return False

    # No Prompt Adapter so far.
    if self.enable_prompt_adapter:
        _raise_or_fallback(feature_name="--enable-prompt-adapter",
                           recommend_to_remove=False)
        return False

    # No text embedding inputs so far.
    if self.enable_prompt_embeds:
        _raise_or_fallback(feature_name="--enable-prompt-embeds",
                           recommend_to_remove=False)
        return False

    # Only Fp16 and Bf16 dtypes since we only support FA.
    V1_SUPPORTED_DTYPES = [torch.bfloat16, torch.float16]
    if model_config.dtype not in V1_SUPPORTED_DTYPES:
        _raise_or_fallback(feature_name=f"--dtype {model_config.dtype}",
                           recommend_to_remove=False)
        return False

    # No Embedding Models so far.
    if model_config.task not in ["generate"]:
        _raise_or_fallback(feature_name=f"--task {model_config.task}",
                           recommend_to_remove=False)
        return False

    # No Mamba or Encoder-Decoder so far.
    if not model_config.is_v1_compatible:
        _raise_or_fallback(feature_name=model_config.architectures,
                           recommend_to_remove=False)
        return False

    # No Concurrent Partial Prefills so far.
    if (self.max_num_partial_prefills
            != SchedulerConfig.max_num_partial_prefills
            or self.max_long_partial_prefills
            != SchedulerConfig.max_long_partial_prefills):
        _raise_or_fallback(feature_name="Concurrent Partial Prefill",
                           recommend_to_remove=False)
        return False

    # No OTLP observability so far.
    if (self.otlp_traces_endpoint or self.collect_detailed_traces):
        _raise_or_fallback(feature_name="--otlp-traces-endpoint",
                           recommend_to_remove=False)
        return False

    # V1 supports N-gram, Medusa, and Eagle speculative decoding.
    is_ngram_enabled = False
    is_eagle_enabled = False
    is_medusa_enabled = False
    if self.speculative_config is not None:
        # This is supported but experimental (handled below).
        speculative_method = self.speculative_config.get("method")
        if speculative_method:
            if speculative_method in ("ngram", "[ngram]"):
                is_ngram_enabled = True
            elif speculative_method == "medusa":
                is_medusa_enabled = True
            elif speculative_method in ("eagle", "eagle3", "deepseek_mtp"):
                is_eagle_enabled = True
        else:
            speculative_model = self.speculative_config.get("model")
            if speculative_model in ("ngram", "[ngram]"):
                is_ngram_enabled = True
        if not (is_ngram_enabled or is_eagle_enabled or is_medusa_enabled):
            # Other speculative decoding methods are not supported yet.
            _raise_or_fallback(feature_name="Speculative Decoding",
                               recommend_to_remove=False)
            return False

    # No XFormers so far.
    V1_BACKENDS = [
        "FLASH_ATTN_VLLM_V1",
        "FLASH_ATTN",
        "PALLAS",
        "PALLAS_VLLM_V1",
        "TRITON_ATTN_VLLM_V1",
        "TRITON_MLA",
        "CUTLASS_MLA_VLLM_V1",
        "FLASHMLA",
        "FLASHINFER",
        "FLASHINFER_VLLM_V1",
        "ROCM_AITER_MLA",
        "TORCH_SDPA_VLLM_V1",
        "FLEX_ATTENTION",
    ]
    if (envs.is_set("VLLM_ATTENTION_BACKEND")
            and envs.VLLM_ATTENTION_BACKEND not in V1_BACKENDS):
        name = f"VLLM_ATTENTION_BACKEND={envs.VLLM_ATTENTION_BACKEND}"
        _raise_or_fallback(feature_name=name, recommend_to_remove=True)
        return False

    # Platforms must decide if they can support v1 for this model
    if not current_platform.supports_v1(model_config=model_config):
        _raise_or_fallback(
            feature_name=f"device type={current_platform.device_type}",
            recommend_to_remove=False)
        return False
    #############################################################
    # Experimental Features - allow users to opt in.

    # Signal Handlers requires running in main thread.
    if (threading.current_thread() != threading.main_thread()
            and _warn_or_fallback("Engine in background thread")):
        return False

    if (self.pipeline_parallel_size > 1 and self.distributed_executor_backend
            not in (ParallelConfig.distributed_executor_backend, "ray", "mp",
                    "external_launcher")):
        name = "Pipeline Parallelism without Ray distributed executor " \
                "or multiprocessing executor or external launcher"
        _raise_or_fallback(feature_name=name, recommend_to_remove=False)
        return False
    #############################################################

    return True


def _set_default_args_v1(self, usage_context: UsageContext) -> None:
    """Set Default Arguments for V1 Engine."""

    # V1 always uses chunked prefills.
    self.enable_chunked_prefill = True

    # V1 enables prefix caching by default.
    if self.enable_prefix_caching is None:
        self.enable_prefix_caching = True

    # V1 should use the new scheduler by default.
    # Swap it only if this arg is set to the original V0 default
    if self.scheduler_cls == EngineArgs.scheduler_cls:
        self.scheduler_cls = "vllm.v1.core.sched.scheduler.Scheduler"

    # vllm-mindspore: Get device memory will initialize device runtime, which
    # will be inherited by the child process in fork mode, resulting in
    # setting device failure for latter ASCEND_RT_VISIBLE_DEVICES modification.
    # So skip it.

    default_max_num_batched_tokens = {
        UsageContext.LLM_CLASS: 8192,
        UsageContext.OPENAI_API_SERVER: 2048,
    }
    default_max_num_seqs = 256

    use_context_value = usage_context.value if usage_context else None
    if (self.max_num_batched_tokens is None
            and usage_context in default_max_num_batched_tokens):
        self.max_num_batched_tokens = default_max_num_batched_tokens[
            usage_context]
        logger.debug(
            "Setting max_num_batched_tokens to %d for %s usage context.",
            self.max_num_batched_tokens, use_context_value)

    if self.max_num_seqs is None:
        self.max_num_seqs = default_max_num_seqs

        logger.debug("Setting max_num_seqs to %d for %s usage context.",
                     self.max_num_seqs, use_context_value)
