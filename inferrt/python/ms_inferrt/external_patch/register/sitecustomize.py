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

"""Auto-load external InferRT vLLM patch in worker subprocesses."""

# sitecustomize must tolerate partially installed optional dependencies and
# therefore performs guarded lazy imports during Python startup.
# pylint: disable=missing-function-docstring,import-outside-toplevel
# pylint: disable=broad-exception-caught,consider-using-from-import,unused-import

from __future__ import annotations

import os


def _fix_triton_constexpr_function() -> None:
    try:
        import triton
        import triton.runtime.jit as triton_jit
    except Exception:
        return

    constexpr_function = getattr(triton_jit, "constexpr_function", None)
    if constexpr_function is None:
        def constexpr_function(fn):
            fn.__triton_builtin__ = True
            return fn

        triton_jit.constexpr_function = constexpr_function

    if not hasattr(triton, "constexpr_function"):
        triton.constexpr_function = constexpr_function


def _fix_torch_npu_synchronize_rule() -> None:
    try:
        import torch._dynamo.trace_rules as trace_rules
        import torch_npu  # noqa: F401 - importing installs NPU Dynamo rules
    except Exception:
        return

    removed = 0
    for rule_map in trace_rules.torch_name_rule_map:
        if rule_map.pop("torch_npu.npu.utils.synchronize", None) is not None:
            removed += 1
    if removed:
        trace_rules.clear_lru_cache()


_fix_triton_constexpr_function()

# Importing torch_npu from sitecustomize also runs in torch-npu profiler
# parser/export subprocesses. Those helper processes only need to parse
# profiling data; eager torch_npu initialization there can deadlock or abort
# Python startup. Keep the workaround opt-in for service runs that need it.
if os.environ.get("MS_INFERRT_FIX_TORCH_NPU_SYNC_RULE", "0") == "1":
    _fix_torch_npu_synchronize_rule()

if os.environ.get("INFERRT_VLLM_EXTERNAL_PATCH", "") == "1":
    try:
        from inferrt_patch import apply_patch

        apply_patch()
    except Exception as exc:
        print(f"Failed to apply external InferRT vLLM patch: {exc}")
