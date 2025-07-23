# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/engine/arg_utils.py
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

import argparse
import json
import threading
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, Literal, get_origin

import torch
import vllm.envs as envs
from pydantic import TypeAdapter, ValidationError
from vllm.config import (ConfigType, GuidedDecodingBackendV1, LoadFormat,
                         ModelConfig, ParallelConfig, SchedulerConfig)
from vllm.engine.arg_utils import (EngineArgs, TypeHint, _raise_or_fallback,
                                   _warn_or_fallback, contains_type, get_args,
                                   get_attr_docs, get_type, get_type_hints,
                                   human_readable_int, is_not_builtin,
                                   literal_to_kwargs, optional_type,
                                   parse_type, union_dict_and_str)

from vllm_mindspore.model_executor.layers.quantization import (
    QUANTIZATION_METHODS)


def get_kwargs(cls: ConfigType) -> dict[str, Any]:
    cls_docs = get_attr_docs(cls)
    kwargs = {}
    for field in fields(cls):
        type_hints: set[TypeHint] = get_type_hints(field.type)

        # If the field is a dataclass, we can use the model_validate_json
        generator = (th for th in type_hints if is_dataclass(th))
        dataclass_cls = next(generator, None)

        # Get the default value of the field
        if field.default is not MISSING:
            default = field.default
        elif field.default_factory is not MISSING:
            default = field.default_factory()

        # Get the help text for the field
        name = field.name
        help = cls_docs[name].strip()
        # Escape % for argparse
        help = help.replace("%", "%%")

        # Initialise the kwargs dictionary for the field
        kwargs[name] = {"default": default, "help": help}

        # Set other kwargs based on the type hints
        json_tip = """\n\nShould either be a valid JSON string or JSON keys
        passed individually. For example, the following sets of arguments are
        equivalent:\n\n
        - `--json-arg '{"key1": "value1", "key2": {"key3": "value2"}}'`\n
        - `--json-arg.key1 value1 --json-arg.key2.key3 value2`\n\n"""
        if dataclass_cls is not None:

            def parse_dataclass(val: str, cls=dataclass_cls) -> Any:
                try:
                    if hasattr(cls, "from_cli"):
                        return cls.from_cli(val)
                    return TypeAdapter(cls).validate_json(val)
                except ValidationError as e:
                    raise argparse.ArgumentTypeError(repr(e)) from e

            kwargs[name]["type"] = parse_dataclass
            kwargs[name]["help"] += json_tip
        elif contains_type(type_hints, bool):
            # Creates --no-<name> and --<name> flags
            kwargs[name]["action"] = argparse.BooleanOptionalAction
        elif contains_type(type_hints, Literal):
            kwargs[name].update(literal_to_kwargs(type_hints))
        elif contains_type(type_hints, tuple):
            type_hint = get_type(type_hints, tuple)
            types = get_args(type_hint)
            tuple_type = types[0]
            assert all(t is tuple_type for t in types if t is not Ellipsis), (
                "All non-Ellipsis tuple elements must be of the same "
                f"type. Got {types}.")
            kwargs[name]["type"] = tuple_type
            kwargs[name]["nargs"] = "+" if Ellipsis in types else len(types)
        elif contains_type(type_hints, list):
            type_hint = get_type(type_hints, list)
            types = get_args(type_hint)
            assert len(types) == 1, (
                "List type must have exactly one type. Got "
                f"{type_hint} with types {types}")
            kwargs[name]["type"] = types[0]
            kwargs[name]["nargs"] = "+"
        elif contains_type(type_hints, int):
            kwargs[name]["type"] = int
            # Special case for large integers
            if name in {"max_model_len", "max_num_batched_tokens"}:
                kwargs[name]["type"] = human_readable_int
        elif contains_type(type_hints, float):
            kwargs[name]["type"] = float
        elif (contains_type(type_hints, dict)
              and (contains_type(type_hints, str)
                   or any(is_not_builtin(th) for th in type_hints))):
            kwargs[name]["type"] = union_dict_and_str
        elif contains_type(type_hints, dict):
            kwargs[name]["type"] = parse_type(json.loads)
            kwargs[name]["help"] += json_tip
        elif (contains_type(type_hints, str)
              or any(is_not_builtin(th) for th in type_hints)):
            kwargs[name]["type"] = str
        else:
            raise ValueError(
                f"Unsupported type {type_hints} for argument {name}.")

        # If the type hint was a sequence of literals, use the helper function
        # to update the type and choices
        if get_origin(kwargs[name].get("type")) is Literal:
            kwargs[name].update(literal_to_kwargs({kwargs[name]["type"]}))

        # If None is in type_hints, make the argument optional.
        # But not if it's a bool, argparse will handle this better.
        if type(None) in type_hints and not contains_type(type_hints, bool):
            kwargs[name]["type"] = optional_type(kwargs[name]["type"])
            if kwargs[name].get("choices"):
                kwargs[name]["choices"].append("None")
        if field.name == "quantization":
            kwargs[name]["choices"] = QUANTIZATION_METHODS
    return kwargs


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
