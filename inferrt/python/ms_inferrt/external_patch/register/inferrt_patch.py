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

"""External InferRT monkey patch for vLLM.

This module is intentionally outside site-packages. It can be loaded manually
by a launcher and automatically in child processes through sitecustomize.
"""

# This compatibility plugin patches optional vLLM/vLLM-Ascend modules at
# runtime, so imports must stay lazy and failures must leave vanilla vLLM
# usable. Marker attributes also intentionally touch third-party classes.
# pylint: disable=missing-function-docstring,import-outside-toplevel
# pylint: disable=broad-exception-caught,protected-access,unused-argument
# pylint: disable=consider-using-from-import

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any


PATCH_DIR = Path(__file__).resolve().parent
WARMUP_ROOT = Path(
    os.environ.get("MS_INFERRT_WARMUP_ROOT", str(PATCH_DIR.parent / "warmup"))
)
RUNTIME_DIR = WARMUP_ROOT / "runtime"


def _enabled() -> bool:
    return os.environ.get("VLLM_TORCH_COMPILE_BACKEND", "").lower() == "inferrt"


def _env_enabled(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    return value not in ("", "0", "false", "False", "FALSE")


def _log(message: str, *args: Any) -> None:
    try:
        from vllm.logger import init_logger

        init_logger("inferrt_vllm_external_patch").info(message, *args)
    except Exception:
        print(message % args if args else message)


def _target_text(target: Any) -> str:
    if isinstance(target, str):
        return target
    qualified_name = getattr(target, "_qualified_op_name", None)
    if qualified_name:
        return qualified_name
    module = getattr(target, "__module__", None)
    name = getattr(target, "__name__", None)
    if module and name:
        return f"{module}.{name}"
    return str(target)


def patch_compilation_backend() -> None:
    from vllm.config.compilation import CompilationConfig

    if getattr(CompilationConfig.init_backend, "_inferrt_external_patched", False):
        return

    original_init_backend = CompilationConfig.init_backend

    def init_backend(self: Any, vllm_config: Any) -> Any:
        if _enabled():
            parallel_config = getattr(vllm_config, "parallel_config", None)
            tensor_parallel_size = int(
                getattr(parallel_config, "tensor_parallel_size", 1) or 1
            )
            disable_tp_ge = os.environ.get(
                "MS_INFERRT_DISABLE_FOR_TP_GE", ""
            ).strip()
            if disable_tp_ge and not _env_enabled("MS_INFERRT_ALLOW_UNSAFE_TP", "0"):
                try:
                    disable_threshold = int(disable_tp_ge)
                except ValueError:
                    disable_threshold = 0
                if 0 < disable_threshold <= tensor_parallel_size:
                    _log(
                        "Bypass InferRT backend for tensor_parallel_size=%s "
                        "because MS_INFERRT_DISABLE_FOR_TP_GE=%s. Falling back "
                        "to vLLM's original backend for correctness.",
                        tensor_parallel_size,
                        disable_threshold,
                    )
                    return original_init_backend(self, vllm_config)

            from ms_inferrt.torch import backend as inferrt_backend
            from backend_optimizer import make_backend

            _log(
                "Using optimized ms_inferrt.torch.backend as vLLM torch.compile backend."
            )
            return make_backend(inferrt_backend)
        return original_init_backend(self, vllm_config)

    init_backend._inferrt_external_patched = True  # type: ignore[attr-defined]
    CompilationConfig.init_backend = init_backend


def patch_torch_compile_options() -> None:
    if not _enabled():
        return

    requested_fullgraph = os.environ.get("MS_INFERRT_DIRECT_FULLGRAPH", "").strip()
    if requested_fullgraph not in ("0", "false", "False", "FALSE"):
        return

    try:
        import torch
    except Exception as exc:
        _log("Skip torch.compile option patch: %s", exc)
        return

    current = torch.compile
    if getattr(current, "_inferrt_external_compile_options_patched", False):
        return

    def compile_with_inferrt_options(*args: Any, **kwargs: Any) -> Any:
        backend = kwargs.get("backend", None)
        is_inferrt_backend = getattr(
            backend, "_ms_inferrt_external_backend", False
        )
        if is_inferrt_backend and kwargs.get("fullgraph", None) is True:
            kwargs = dict(kwargs)
            kwargs["fullgraph"] = False
            _log(
                "Changed torch.compile fullgraph=True to fullgraph=False for "
                "the direct InferRT backend because MS_INFERRT_DIRECT_FULLGRAPH=0."
            )
        return current(*args, **kwargs)

    compile_with_inferrt_options._inferrt_external_compile_options_patched = True  # type: ignore[attr-defined]
    torch.compile = compile_with_inferrt_options


def patch_inferrt_disabled_ops() -> None:
    patterns = [
        item.strip()
        for item in os.environ.get("MS_INFERRT_DISABLE_OP_PATTERNS", "").split(",")
        if item.strip()
    ]
    if _env_enabled("MS_INFERRT_DISABLE_TP_COMM_IN_GRAPH", "1"):
        for item in (
            "vllm::all_reduce",
            "maybe_all_reduce_tensor_model_parallel",
            "_c10d_functional.all_reduce",
        ):
            if item not in patterns:
                patterns.append(item)
    force_vllm_python_call = _env_enabled(
        "MS_INFERRT_FORCE_VLLM_OPS_PYTHON_CALL", "0"
    )
    if not patterns and not force_vllm_python_call:
        return

    try:
        import ms_inferrt.torch.fx_backend as fx_backend
    except Exception as exc:
        _log("Skip InferRT op-disable patch: %s", exc)
        return

    op_map = getattr(fx_backend, "_OP_MAP", None)
    if patterns and not isinstance(op_map, dict):
        _log("Skip InferRT op-disable patch: _OP_MAP not found.")
        return

    removed: list[str] = []
    if patterns:
        for key in list(op_map.keys()):
            key_text = _target_text(key)
            if any(pattern in key_text for pattern in patterns):
                removed.append(key_text)
                op_map.pop(key, None)

        _log(
            "Disabled %d InferRT op mappings by patterns %s: %s",
            len(removed),
            patterns,
            removed[:20],
        )

    original_get_op = getattr(fx_backend, "_get_op", None)
    if original_get_op is None or getattr(
        original_get_op, "_inferrt_external_disabled_ops_patched", False
    ):
        return

    def get_op_with_disabled_patterns(target: Any) -> Any:
        target_text = _target_text(target)
        target_name = getattr(target, "__name__", "")
        target_module = getattr(target, "__module__", "")
        if (
            _env_enabled("MS_INFERRT_DISABLE_TP_COMM_IN_GRAPH", "1")
            and (
                target_text in ("vllm::all_reduce", "vllm::maybe_all_reduce_tensor_model_parallel")
                or target_name in ("all_reduce", "maybe_all_reduce_tensor_model_parallel")
                or "maybe_all_reduce_tensor_model_parallel" in target_text
                or "_c10d_functional.all_reduce" in target_text
            )
        ):
            return fx_backend.Op.python_call
        if force_vllm_python_call and (
            "vllm::" in target_text
            or target_module.startswith("torch._ops.vllm")
            or "torch.ops.vllm" in target_text
        ):
            return fx_backend.Op.python_call
        if any(
            pattern in target_text or (target_name and pattern in target_name)
            for pattern in patterns
        ):
            return fx_backend.Op.custom_call
        return original_get_op(target)

    get_op_with_disabled_patterns._inferrt_external_disabled_ops_patched = True  # type: ignore[attr-defined]
    fx_backend._get_op = get_op_with_disabled_patterns
    _log(
        "Patched InferRT _get_op: disabled patterns=%s, force_vllm_python_call=%s",
        patterns,
        force_vllm_python_call,
    )


def patch_inferrt_tp_reduce_identity_guard() -> None:
    if not _enabled():
        return

    try:
        import ms_inferrt.torch.fx_backend as fx_backend
    except Exception as exc:
        _log("Skip InferRT TP reduce identity guard: %s", exc)
        return

    original = getattr(fx_backend, "_try_handle_vllm_identity_control_op", None)
    if original is None or getattr(
        original, "_inferrt_external_tp_reduce_guard", False
    ):
        return

    def guarded_identity_control_op(node: Any, env: dict[Any, Any]) -> bool:
        target_text = _target_text(getattr(node, "target", None))
        try:
            tp_size = int(os.environ.get("MS_INFERRT_TENSOR_PARALLEL_SIZE", "1"))
        except ValueError:
            tp_size = 1
        if (
            tp_size > 1
            and not _env_enabled("MS_INFERRT_UNSAFE_PRUNE_TP_REDUCE", "0")
            and target_text.endswith("maybe_pad_and_reduce")
        ):
            return False
        return original(node, env)

    guarded_identity_control_op._inferrt_external_tp_reduce_guard = True  # type: ignore[attr-defined]
    fx_backend._try_handle_vllm_identity_control_op = guarded_identity_control_op
    _log(
        "Guarded InferRT vLLM control lowering: maybe_pad_and_reduce will not "
        "be lowered to identity when tensor_parallel_size>1.",
    )


def patch_npu_worker_zero_kv_memory_guard() -> None:
    if not _enabled() or not _env_enabled("MS_INFERRT_PATCH_ZERO_KV_MEMORY", "1"):
        return

    try:
        import torch
        from vllm_ascend.worker.worker import NPUWorker
    except Exception as exc:
        _log("Skip NPUWorker zero KV memory guard: %s", exc)
        return

    current = getattr(NPUWorker, "determine_available_memory", None)
    if current is None or getattr(current, "_inferrt_zero_kv_memory_guard", False):
        return

    def determine_available_memory(self: Any) -> int:
        available_memory = int(current(self))
        if available_memory > 0:
            return available_memory

        try:
            free_memory, total_memory = torch.npu.mem_get_info()
            utilization = float(getattr(self.cache_config, "gpu_memory_utilization", 0.9))
            requested_memory = int(total_memory * utilization)
            used_memory = max(int(total_memory) - int(free_memory), 0)
            try:
                safety_gib = float(
                    os.environ.get("MS_INFERRT_KV_MEMORY_FALLBACK_SAFETY_GB", "2")
                )
            except ValueError:
                safety_gib = 2.0
            safety_bytes = int(safety_gib * 1024**3)
            fallback_memory = max(requested_memory - used_memory - safety_bytes, 0)
            min_gib = float(
                os.environ.get("MS_INFERRT_KV_MEMORY_FALLBACK_MIN_GB", "1")
            )
            min_bytes = int(min_gib * 1024**3)
            if fallback_memory < min_bytes and free_memory > safety_bytes + min_bytes:
                fallback_memory = min_bytes
        except Exception as exc:
            _log("Failed to estimate fallback KV memory after zero result: %s", exc)
            return available_memory

        if fallback_memory <= 0:
            _log(
                "NPUWorker determine_available_memory returned 0 and fallback "
                "estimate is also 0: free=%s total=%s.",
                free_memory,
                total_memory,
            )
            return available_memory

        _log(
            "NPUWorker determine_available_memory returned 0 after InferRT "
            "profile. Use fallback KV cache memory %d bytes "
            "(free=%d, total=%d, utilization=%.3f, safety_gb=%.2f).",
            fallback_memory,
            free_memory,
            total_memory,
            utilization,
            safety_gib,
        )
        return int(fallback_memory)

    determine_available_memory._inferrt_zero_kv_memory_guard = True  # type: ignore[attr-defined]
    NPUWorker.determine_available_memory = determine_available_memory
    _log("Patched NPUWorker zero KV memory guard for direct InferRT backend.")


def _configure_decode_cudagraph(compilation_config: Any, cudagraph_mode: Any) -> bool:
    """Select native decode ACLGraph without enabling it for prefill."""
    decode_original_backend = _env_enabled(
        "MS_INFERRT_DECODE_USE_ORIGINAL_BACKEND", "1"
    )
    if decode_original_backend:
        capture_sizes_text = os.environ.get(
            "MS_INFERRT_DECODE_CUDAGRAPH_CAPTURE_SIZES",
            "1,2,4,8,16,32,64,128,256",
        )
        capture_sizes = [
            int(item.strip())
            for item in capture_sizes_text.split(",")
            if item.strip()
        ] or [1]
        compilation_config.cudagraph_mode = cudagraph_mode.FULL_DECODE_ONLY
        compilation_config.cudagraph_capture_sizes = capture_sizes
        compilation_config.max_cudagraph_capture_size = max(capture_sizes)
    elif not _env_enabled("MS_INFERRT_KEEP_VLLM_CUDAGRAPH", "0"):
        compilation_config.cudagraph_mode = cudagraph_mode.NONE
        compilation_config.cudagraph_capture_sizes = []
        compilation_config.max_cudagraph_capture_size = 0
    return decode_original_backend


def patch_vllm_ascend_config() -> None:
    try:
        from vllm_ascend.platform import NPUPlatform
    except Exception:
        return

    current = NPUPlatform.check_and_update_config
    if getattr(current, "_inferrt_external_patched", False):
        return

    original_check_and_update_config = current

    def check_and_update_config(cls: Any, vllm_config: Any) -> None:
        original_check_and_update_config(vllm_config)

        if not _enabled():
            return

        from vllm.config import CUDAGraphMode, CompilationMode

        compilation_config = vllm_config.compilation_config
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        parallel_config = getattr(vllm_config, "parallel_config", None)
        tensor_parallel_size = int(
            getattr(parallel_config, "tensor_parallel_size", 1) or 1
        )
        os.environ["MS_INFERRT_TENSOR_PARALLEL_SIZE"] = str(tensor_parallel_size)
        max_num_batched_tokens = getattr(
            scheduler_config, "max_num_batched_tokens", None
        )
        if max_num_batched_tokens is not None:
            os.environ["MS_INFERRT_VLLM_MAX_NUM_BATCHED_TOKENS"] = str(
                max_num_batched_tokens
            )

        if not _env_enabled("MS_INFERRT_KEEP_VLLM_COMPILATION_MODE", "0"):
            compilation_config.mode = CompilationMode.DYNAMO_TRACE_ONCE
        decode_original_backend = _configure_decode_cudagraph(
            compilation_config, CUDAGraphMode
        )
        custom_ops_override = os.environ.get(
            "MS_INFERRT_VLLM_CUSTOM_OPS_OVERRIDE", ""
        ).strip()
        if custom_ops_override:
            compilation_config.custom_ops = [
                item.strip()
                for item in custom_ops_override.split(",")
                if item.strip()
            ]
        if _env_enabled("MS_INFERRT_CLEAR_SPLITTING_OPS", "0"):
            compilation_config.splitting_ops = []
        elif compilation_config.splitting_ops is None:
            compilation_config.splitting_ops = []
        splitting_ops_override = os.environ.get(
            "MS_INFERRT_VLLM_SPLITTING_OPS_OVERRIDE", ""
        ).strip()
        if splitting_ops_override:
            compilation_config.splitting_ops = [
                item.strip()
                for item in splitting_ops_override.split(",")
                if item.strip()
            ]

        try:
            from vllm_ascend.ascend_config import init_ascend_config

            if (
                not decode_original_backend
                and not _env_enabled("MS_INFERRT_KEEP_VLLM_CUDAGRAPH", "0")
            ):
                init_ascend_config(vllm_config).enable_npugraph_ex = False
        except Exception:
            pass

        _log(
            "Configured direct InferRT torch.compile backend with mode %s "
            "and cudagraph %s.",
            compilation_config.mode,
            compilation_config.cudagraph_mode,
        )

    check_and_update_config._inferrt_external_patched = True  # type: ignore[attr-defined]
    NPUPlatform.check_and_update_config = classmethod(check_and_update_config)


def _force_direct_inferrt_compilation(vllm_config: Any, source: str) -> None:
    """Force vLLM compilation config onto the direct torch.compile path.

    vllm-mindspore currently sets ``use_inductor=False`` and defaults to
    ``CompilationLevel.PIECEWISE`` in its config post-init. On Ascend this is
    later interpreted as ACLGraph/piecewise execution, even if the user passed
    ``backend=inferrt``. For the external InferRT backend experiment, keep the
    backend at the torch.compile/Inductor-level replacement boundary instead.
    """

    if not _enabled() or _env_enabled("MS_INFERRT_KEEP_MS_COMPILE_MODE", "0"):
        return

    try:
        from vllm.config import CompilationMode, CUDAGraphMode
    except Exception as exc:
        _log("Skip direct InferRT compilation config patch from %s: %s", source, exc)
        return

    compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is None:
        return

    decode_capture_sizes_text = os.environ.get(
        "MS_INFERRT_DECODE_CUDAGRAPH_CAPTURE_SIZES",
        "1,2,4,8,16,32,64,128,256",
    )
    decode_capture_sizes = [
        int(item.strip())
        for item in decode_capture_sizes_text.split(",")
        if item.strip()
    ]
    if not decode_capture_sizes:
        decode_capture_sizes = [1]

    compilation_config.backend = "inferrt"
    compilation_config.level = CompilationMode.DYNAMO_TRACE_ONCE
    compilation_config.mode = CompilationMode.DYNAMO_TRACE_ONCE
    if _env_enabled("MS_INFERRT_DECODE_USE_ORIGINAL_BACKEND", "1"):
        # Keep mixed/prefill execution out of vLLM piecewise/ACLGraph while
        # allowing decode to use the platform's normal full-graph path.  This
        # preserves the direct InferRT backend boundary for prefill and lets
        # decode run the original vLLM fast path.
        compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        compilation_config.use_cudagraph = True
        compilation_config.cudagraph_capture_sizes = decode_capture_sizes
        compilation_config.max_cudagraph_capture_size = max(decode_capture_sizes)
        compilation_config.cudagraph_num_of_warmups = int(
            os.environ.get("MS_INFERRT_DECODE_CUDAGRAPH_WARMUPS", "1")
        )
    else:
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        compilation_config.cudagraph_num_of_warmups = 0
        compilation_config.cudagraph_capture_sizes = []
        compilation_config.max_cudagraph_capture_size = 0
        compilation_config.use_cudagraph = False
    compilation_config.compile_sizes = []
    compilation_config.splitting_ops = []
    compilation_config.use_inductor = True
    compilation_config.use_inductor_graph_partition = False
    custom_ops_override = os.environ.get(
        "MS_INFERRT_VLLM_CUSTOM_OPS_OVERRIDE", ""
    ).strip()
    if custom_ops_override:
        compilation_config.custom_ops = [
            item.strip()
            for item in custom_ops_override.split(",")
            if item.strip()
        ]
    else:
        # Keep vLLM custom ops available while still forcing the outer
        # compilation boundary to the direct InferRT torch.compile backend.
        # Forcing ["none"] here makes operators such as attention/norm take
        # unfriendly FX fallback paths and can break request execution.
        compilation_config.custom_ops = ["all"]

    try:
        compilation_config.pass_config.enable_fusion = False
        compilation_config.pass_config.enable_noop = False
    except Exception:
        pass

    _log(
        "Forced vllm-mindspore direct InferRT compilation config from %s: "
        "mode=%s, cudagraph=%s, backend=%s.",
        source,
        compilation_config.mode,
        compilation_config.cudagraph_mode,
        compilation_config.backend,
    )


def patch_vllm_mindspore_config() -> None:
    """Patch the actual MindSpore platform package used in this environment."""

    try:
        from vllm.platforms import current_platform
    except Exception as exc:
        _log("Skip current platform config patch: %s", exc)
        return

    current = getattr(current_platform, "check_and_update_config", None)
    if current is None:
        _log("Skip current platform config patch: check_and_update_config not found.")
        return
    if getattr(current, "_inferrt_external_ms_patched", False):
        return

    original_check_and_update_config = current

    def check_and_update_config(vllm_config: Any) -> None:
        # vllm_mindspore.config.vllm_config_post_init has already set the
        # default to PIECEWISE/use_inductor=False before this hook runs. Flip
        # it before the original platform check so it cannot enable ACLGraph.
        _force_direct_inferrt_compilation(vllm_config, "before_ms_platform_check")
        original_check_and_update_config(vllm_config)
        # Keep it pinned after the platform check too, because the original
        # code may mutate splitting_ops or cudagraph settings.
        _force_direct_inferrt_compilation(vllm_config, "after_ms_platform_check")

    check_and_update_config._inferrt_external_ms_patched = True  # type: ignore[attr-defined]
    setattr(current_platform, "check_and_update_config", check_and_update_config)


def _stage_from_attention_metadata(attn_metadata: Any) -> str:
    """Infer the current vLLM execution stage from attention metadata.

    This is deliberately conservative for prefill: any mixed or unknown case is
    treated as prefill/unknown so it still goes through InferRT. Only a clear
    decode-only marker is allowed to bypass InferRT when MS_INFERRT_PREFILL_ONLY
    is enabled.
    """

    seen_decode = False
    seen_prefill = False
    seen_unknown = False

    def visit(value: Any) -> None:
        nonlocal seen_decode, seen_prefill, seen_unknown
        if value is None:
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
            return

        num_prefills = getattr(value, "num_prefills", None)
        num_decodes = getattr(value, "num_decodes", None)
        try:
            if num_prefills is not None and int(num_prefills) > 0:
                seen_prefill = True
            if num_decodes is not None and int(num_decodes) > 0:
                seen_decode = True
        except Exception:
            pass

        attn_state = getattr(value, "attn_state", None)
        if attn_state is not None:
            state_name = getattr(attn_state, "name", str(attn_state))
            state_text = str(state_name).lower()
            if "decode" in state_text and "prefill" not in state_text:
                seen_decode = True
            elif "prefill" in state_text:
                seen_prefill = True
            else:
                seen_unknown = True

        max_query_len = getattr(value, "max_query_len", None)
        try:
            if max_query_len is not None and int(max_query_len) > 1:
                seen_prefill = True
        except Exception:
            pass

    visit(attn_metadata)

    if seen_prefill:
        return "prefill"
    if seen_decode and not seen_unknown:
        return "decode"
    return "unknown"


def _make_stage_set_forward_context(original: Any) -> Any:
    if getattr(original, "_inferrt_runtime_stage_patched", False):
        return original

    @contextlib.contextmanager
    def set_forward_context_with_stage(attn_metadata: Any, *args: Any, **kwargs: Any):
        old_stage = os.environ.get("MS_INFERRT_RUNTIME_STAGE")
        stage = _stage_from_attention_metadata(attn_metadata)
        os.environ["MS_INFERRT_RUNTIME_STAGE"] = stage
        try:
            with original(attn_metadata, *args, **kwargs):
                yield
        finally:
            if old_stage is None:
                os.environ.pop("MS_INFERRT_RUNTIME_STAGE", None)
            else:
                os.environ["MS_INFERRT_RUNTIME_STAGE"] = old_stage

    set_forward_context_with_stage._inferrt_runtime_stage_patched = True  # type: ignore[attr-defined]
    return set_forward_context_with_stage


def _set_runtime_stage(stage: str):
    @contextlib.contextmanager
    def manager():
        old_stage = os.environ.get("MS_INFERRT_RUNTIME_STAGE")
        os.environ["MS_INFERRT_RUNTIME_STAGE"] = stage
        try:
            yield
        finally:
            if old_stage is None:
                os.environ.pop("MS_INFERRT_RUNTIME_STAGE", None)
            else:
                os.environ["MS_INFERRT_RUNTIME_STAGE"] = old_stage

    return manager()


def _make_stage_set_ascend_forward_context(original: Any) -> Any:
    if getattr(original, "_inferrt_runtime_stage_patched", False):
        return original

    @contextlib.contextmanager
    def set_ascend_forward_context_with_stage(
        attn_metadata: Any, *args: Any, **kwargs: Any
    ):
        old_stage = os.environ.get("MS_INFERRT_RUNTIME_STAGE")
        stage = _stage_from_attention_metadata(attn_metadata)
        runtime_mode = kwargs.get("aclgraph_runtime_mode", None)
        if runtime_mode is not None:
            mode_text = getattr(runtime_mode, "name", str(runtime_mode)).lower()
            if "full" in mode_text:
                stage = "decode"
        os.environ["MS_INFERRT_RUNTIME_STAGE"] = stage
        try:
            with original(attn_metadata, *args, **kwargs):
                yield
        finally:
            if old_stage is None:
                os.environ.pop("MS_INFERRT_RUNTIME_STAGE", None)
            else:
                os.environ["MS_INFERRT_RUNTIME_STAGE"] = old_stage

    set_ascend_forward_context_with_stage._inferrt_runtime_stage_patched = True  # type: ignore[attr-defined]
    return set_ascend_forward_context_with_stage


def patch_decode_dummy_run_stage_marker() -> None:
    if not _enabled():
        return

    try:
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    except Exception as exc:
        _log("Skip vLLM-Ascend dummy-run stage marker patch: %s", exc)
        return

    current = getattr(NPUModelRunner, "_dummy_run", None)
    if current is None or getattr(current, "_inferrt_runtime_stage_patched", False):
        return

    def dummy_run_with_stage(self: Any, *args: Any, **kwargs: Any):
        uniform_decode = bool(kwargs.get("uniform_decode", False))
        with_prefill = bool(kwargs.get("with_prefill", False))
        outer_stage = os.environ.get("MS_INFERRT_RUNTIME_STAGE", "").lower()
        stage = (
            "decode"
            if outer_stage == "decode" or (uniform_decode and not with_prefill)
            else "prefill"
        )
        with _set_runtime_stage(stage):
            return current(self, *args, **kwargs)

    dummy_run_with_stage._inferrt_runtime_stage_patched = True  # type: ignore[attr-defined]
    NPUModelRunner._dummy_run = dummy_run_with_stage
    _log("Patched vLLM-Ascend NPUModelRunner._dummy_run runtime stage marker.")


def _patch_method_stage_marker(cls: Any, method_name: str, stage: str) -> bool:
    current = getattr(cls, method_name, None)
    if current is None or getattr(current, "_inferrt_runtime_stage_patched", False):
        return False

    def method_with_stage(self: Any, *args: Any, **kwargs: Any):
        with _set_runtime_stage(stage):
            return current(self, *args, **kwargs)

    method_with_stage._inferrt_runtime_stage_patched = True  # type: ignore[attr-defined]
    setattr(cls, method_name, method_with_stage)
    return True


def patch_decode_capture_stage_marker() -> None:
    if not _enabled():
        return

    patched = 0
    try:
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        if _patch_method_stage_marker(NPUModelRunner, "capture_model", "decode"):
            patched += 1
    except Exception as exc:
        _log("Skip vLLM-Ascend capture_model stage marker patch: %s", exc)

    if patched:
        _log(
            "Patched vLLM decode capture_model runtime stage marker "
            "for InferRT prefill-only routing (%d bindings).",
            patched,
        )


def patch_aclgraph_call_stage_marker() -> None:
    """Mark vLLM-Ascend ACLGraph execution as decode for runtime dispatch.

    vLLM may reuse the same torch.compile wrapper compiled during prefill when
    it captures/runs decode ACLGraphs.  The backend returns a prefill-only
    dispatcher, so ACLGraph execution must carry an explicit decode stage.
    """

    if not _enabled():
        return

    try:
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    except Exception as exc:
        _log("Skip vLLM-Ascend ACLGraphWrapper stage marker patch: %s", exc)
        return

    current = getattr(ACLGraphWrapper, "__call__", None)
    if current is None or getattr(current, "_inferrt_runtime_stage_patched", False):
        return

    def ensure_aclgraph_params(self: Any) -> None:
        try:
            from vllm_ascend.compilation import acl_graph

            if acl_graph.get_graph_params() is not None:
                return

            sizes = list(
                getattr(
                    getattr(self, "compilation_config", None),
                    "cudagraph_capture_sizes",
                    [],
                )
                or []
            )
            if not sizes:
                try:
                    from vllm.forward_context import get_forward_context

                    batch_descriptor = getattr(
                        get_forward_context(), "batch_descriptor", None
                    )
                    num_tokens = getattr(batch_descriptor, "num_tokens", None)
                    if num_tokens is not None:
                        sizes = [int(num_tokens)]
                except Exception:
                    sizes = []
            if not sizes:
                sizes = [1]

            acl_graph.set_graph_params(sorted(set(int(size) for size in sizes)))
            _log(
                "Initialized missing vLLM-Ascend ACLGraph GraphParams with "
                "capture sizes %s.",
                sizes,
            )
        except ValueError:
            # Another path initialized graph params after our None check.
            return
        except Exception as exc:
            _log("Skip ACLGraph GraphParams guard: %s", exc)

    def aclgraph_call_with_stage(self: Any, *args: Any, **kwargs: Any):
        with _set_runtime_stage("decode"):
            ensure_aclgraph_params(self)
            return current(self, *args, **kwargs)

    aclgraph_call_with_stage._inferrt_runtime_stage_patched = True  # type: ignore[attr-defined]
    ACLGraphWrapper.__call__ = aclgraph_call_with_stage
    _log(
        "Patched vLLM-Ascend ACLGraphWrapper.__call__ runtime stage marker "
        "for InferRT prefill-only dispatch."
    )


def patch_runtime_stage_marker() -> None:
    """Mark prefill/decode explicitly while vLLM executes the model forward."""

    if not _enabled():
        return

    patched = 0
    try:
        import vllm.forward_context as forward_context

        original = getattr(forward_context, "set_forward_context", None)
        if original is not None:
            wrapped = _make_stage_set_forward_context(original)
            if wrapped is not original:
                forward_context.set_forward_context = wrapped
                patched += 1
    except Exception as exc:
        _log("Skip vLLM forward_context stage marker patch: %s", exc)

    for module_name in (
        "vllm.v1.worker.gpu.model_runner",
        "vllm.v1.worker.gpu_model_runner",
        "vllm_ascend.worker.v2.aclgraph_utils",
    ):
        try:
            module = __import__(module_name, fromlist=["set_forward_context"])
            original = getattr(module, "set_forward_context", None)
            if original is None:
                continue
            wrapped = _make_stage_set_forward_context(original)
            if wrapped is not original:
                setattr(module, "set_forward_context", wrapped)
                patched += 1
        except Exception:
            continue

    try:
        import vllm_ascend.ascend_forward_context as ascend_forward_context

        original = getattr(ascend_forward_context, "set_ascend_forward_context", None)
        if original is not None:
            wrapped = _make_stage_set_ascend_forward_context(original)
            if wrapped is not original:
                ascend_forward_context.set_ascend_forward_context = wrapped
                patched += 1
    except Exception as exc:
        _log("Skip vLLM-Ascend forward_context stage marker patch: %s", exc)

    for module_name in (
        "vllm_ascend.worker.model_runner_v1",
        "vllm_ascend.spec_decode.eagle_proposer",
        "vllm_ascend.spec_decode.medusa_proposer",
    ):
        try:
            module = __import__(module_name, fromlist=["set_ascend_forward_context"])
            original = getattr(module, "set_ascend_forward_context", None)
            if original is None:
                continue
            wrapped = _make_stage_set_ascend_forward_context(original)
            if wrapped is not original:
                setattr(module, "set_ascend_forward_context", wrapped)
                patched += 1
        except Exception:
            continue

    if patched:
        _log(
            "Patched vLLM forward_context runtime stage marker for InferRT "
            "prefill/decode routing (%d bindings).",
            patched,
        )


def apply_patch() -> None:
    os.environ.setdefault("VLLM_TORCH_COMPILE_BACKEND", "inferrt")
    os.environ.setdefault("VLLM_USE_BYTECODE_HOOK", "0")
    os.environ.setdefault("MS_INFERRT_USE_COMPILER_INTERFACE", "0")
    os.environ.setdefault("MS_INFERRT_EXTERNAL_OPT_PROFILE", "auto")
    patch_torch_compile_options()
    patch_compilation_backend()
    patch_inferrt_disabled_ops()
    patch_inferrt_tp_reduce_identity_guard()
    patch_npu_worker_zero_kv_memory_guard()
    patch_vllm_ascend_config()
    patch_vllm_mindspore_config()
    patch_runtime_stage_marker()
    patch_decode_dummy_run_stage_marker()
    patch_decode_capture_stage_marker()
    patch_aclgraph_call_stage_marker()
    try:
        from qwen35_compat import apply_qwen35_compat_patches

        apply_qwen35_compat_patches()
    except Exception as exc:
        _log("Skip Qwen3.5 InferRT compatibility patches: %s", exc)
    try:
        from warmup_patch import patch_vllm_ascend_warmup

        patch_vllm_ascend_warmup()
    except Exception as exc:
        _log("Skip InferRT extra warmup patch: %s", exc)
    _log("External InferRT vLLM monkey patch is enabled.")


__all__ = ["apply_patch"]
