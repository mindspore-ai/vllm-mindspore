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

"""Thin wrapper around the real ms_inferrt torch backend.

This module intentionally avoids copying InferRT FX lowering logic.  The
backend implementation and vLLM-oriented FX handling live in ``ms_inferrt``;
the external patch only records compile signatures so AI warmup can verify that
real requests reuse graphs seen during warmup.
"""

# Dynamic backend adapters intentionally use lazy imports, compatibility
# fallbacks, and marker attributes on objects owned by torch and vLLM.
# pylint: disable=missing-function-docstring,import-outside-toplevel
# pylint: disable=broad-exception-caught,protected-access,unused-argument

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.fx.graph_module import GraphModule


PATCH_DIR = Path(__file__).resolve().parent
WARMUP_ROOT = Path(
    os.environ.get("MS_INFERRT_WARMUP_ROOT", str(PATCH_DIR.parent / "warmup"))
)
RUNTIME_DIR = WARMUP_ROOT / "runtime"

_GRAPH_ID = 0
_PRECOMPILED_GRAPH_SIGNATURES: set[str] = set()


def _enabled(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    return value not in ("", "0", "false", "False", "FALSE")


def _next_graph_id() -> int:
    global _GRAPH_ID
    _GRAPH_ID += 1
    return _GRAPH_ID


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


def _value_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {
            "type": "Tensor",
            "shape": [str(dim) for dim in value.shape],
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (tuple, list)):
        return {
            "type": type(value).__name__,
            "len": len(value),
            "items": [_value_summary(item) for item in value[:4]],
        }
    return {"type": type(value).__name__, "value": str(value)[:120]}


def _report_dir() -> Path:
    return Path(
        os.environ.get(
            "MS_INFERRT_EXTERNAL_OPT_REPORT_DIR",
            os.environ.get(
                "MS_INFERRT_INTERNAL_REPORT_DIR",
                str(RUNTIME_DIR / "inferrt_external_opt_reports"),
            ),
        )
    )


def _graph_stats(gm: GraphModule, example_inputs: Any) -> dict[str, Any]:
    targets = [_target_text(node.target) for node in gm.graph.nodes]
    lowered_targets = [item.lower() for item in targets]
    placeholder_count = sum(1 for node in gm.graph.nodes if node.op == "placeholder")
    max_input_dim = 0
    for item in _iter_values(example_inputs):
        if isinstance(item, torch.Tensor) and item.ndim:
            try:
                max_input_dim = max(max_input_dim, max(int(dim) for dim in item.shape))
            except Exception:
                pass
    return {
        "node_count": len(targets),
        "placeholder_count": placeholder_count,
        "attention_count": sum("attention" in item for item in lowered_targets),
        "matmul_count": sum("matmul" in item or "mm." in item for item in lowered_targets),
        "full_graph_fia_count": sum("full_graph_fia" in item for item in lowered_targets),
        "aclgraph_target_count": sum("aclgraph" in item or "acl_graph" in item for item in lowered_targets),
        "max_input_dim": max_input_dim,
        "target_counts_top": dict(Counter(targets).most_common(40)),
    }


def _should_use_inferrt_for_graph(stats: dict[str, Any]) -> tuple[bool, str]:
    policy = os.environ.get("MS_INFERRT_BACKEND_STAGE_POLICY", "").strip().lower()
    if policy in ("all", "always", "force"):
        return True, "policy_all"
    if policy in ("off", "none", "disable"):
        return False, "policy_off"

    # Default policy: compile shared model graphs with InferRT even if vLLM
    # first captures them while building decode ACLGraphs. Runtime dispatch below
    # still routes decode calls to gm.forward, so decode execution stays native.
    if _enabled("MS_INFERRT_PREFILL_ONLY", "1"):
        stage = os.environ.get("MS_INFERRT_RUNTIME_STAGE", "").strip().lower()
        if stage == "decode" and not _enabled("MS_INFERRT_COMPILE_DECODE", "0"):
            if (
                int(stats.get("full_graph_fia_count", 0) or 0) > 0
                or int(stats.get("aclgraph_target_count", 0) or 0) > 0
            ):
                return False, "decode_aclgraph_target"
            return True, "decode_capture_shared_model_graph"
        if stage:
            return True, f"explicit_{stage}_stage"
        if (
            not _enabled("MS_INFERRT_COMPILE_DECODE", "0")
            and (
                int(stats.get("full_graph_fia_count", 0) or 0) > 0
                or int(stats.get("aclgraph_target_count", 0) or 0) > 0
            )
        ):
            return False, "decode_aclgraph_target"
        return True, "prefill_or_unknown_graph"
    return True, "prefill_only_disabled"


def _compile_decode_backend(gm: GraphModule, example_inputs: Any) -> Any:
    backend_name = os.environ.get("MS_INFERRT_DECODE_BACKEND", "eager").strip()
    if backend_name in ("", "eager", "none"):
        return gm.forward

    try:
        from torch._dynamo.backends.registry import lookup_backend

        backend = lookup_backend(backend_name)
        return backend(gm, example_inputs)
    except Exception as exc:
        if _enabled("MS_INFERRT_STRICT_DECODE_BACKEND", "0"):
            raise
        print(
            "InferRT decode backend fallback failed for "
            f"MS_INFERRT_DECODE_BACKEND={backend_name!r}: {exc}. "
            "Use gm.forward instead.",
            flush=True,
        )
        return gm.forward


def _runtime_dispatch_prefill_only(gm: GraphModule, compiled_callable: Any) -> Any:
    """Dispatch a compiled model region by runtime stage.

    vLLM uses the same torch.compile wrapper for both prefill and decode.  A
    graph first compiled during prefill would otherwise be reused by decode
    ACLGraph capture, which makes decode execute InferRT as well.  Keep the
    prefill compiled callable, but call the original FX ``gm.forward`` when a
    runtime stage marker explicitly says decode.
    """

    if not _enabled("MS_INFERRT_PREFILL_ONLY", "1"):
        return compiled_callable
    if _enabled("MS_INFERRT_COMPILE_DECODE", "0"):
        return compiled_callable

    def callable_with_stage_dispatch(*args: Any, **kwargs: Any) -> Any:
        stage = os.environ.get("MS_INFERRT_RUNTIME_STAGE", "").strip().lower()
        if stage == "decode":
            return gm.forward(*args, **kwargs)
        return compiled_callable(*args, **kwargs)

    callable_with_stage_dispatch._ms_inferrt_prefill_only_dispatch = True  # type: ignore[attr-defined]
    return callable_with_stage_dispatch


def _iter_values(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_values(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_values(item)
    else:
        yield value


def _contains_unsupported_inferrt_input(value: Any) -> str | None:
    if isinstance(value, (immutable_dict, dict)):
        for item in value.values():
            reason = _contains_unsupported_inferrt_input(item)
            if reason:
                return reason
        return None
    if isinstance(value, (immutable_list, tuple, list)):
        for item in value:
            reason = _contains_unsupported_inferrt_input(item)
            if reason:
                return reason
    return None


def _graph_signature(gm: GraphModule) -> dict[str, Any]:
    placeholders: list[dict[str, Any]] = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            placeholders.append(_value_summary(node.meta.get("example_value", None)))

    targets = [_target_text(node.target) for node in gm.graph.nodes]
    payload = {
        "placeholders": placeholders,
        "target_counts": dict(Counter(targets).most_common()),
        "node_count": len(targets),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {"hash": hashlib.sha256(encoded).hexdigest()[:24], "payload": payload}


def _append_compile_cache_report(
    graph_id: int,
    signature: dict[str, Any],
    stats: dict[str, Any],
    cache_event: str,
    phase: str,
    bucket: str,
) -> None:
    if not (
        _enabled("MS_INFERRT_GRAPH_CACHE_REPORT", "1")
        or _enabled("MS_INFERRT_INTERNAL_REPORT", "0")
    ):
        return
    out_dir = _report_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"graph_compile_cache_pid{os.getpid()}.jsonl"
    payload = {
        "graph_id": graph_id,
        "pid": os.getpid(),
        "signature_hash": signature["hash"],
        "cache_event": cache_event,
        "warmup_phase": phase,
        "warmup_bucket": bucket,
        "warmup_completed": os.environ.get("MS_INFERRT_WARMUP_COMPLETED", "0"),
        "stats": {
            "node_count": stats.get("node_count"),
            "placeholder_count": stats.get("placeholder_count"),
            "attention_count": stats.get("attention_count"),
            "matmul_count": stats.get("matmul_count"),
            "full_graph_fia_count": stats.get("full_graph_fia_count"),
            "aclgraph_target_count": stats.get("aclgraph_target_count"),
            "max_input_dim": stats.get("max_input_dim"),
            "resolved_profile": stats.get("resolved_profile"),
            "backend_decision": stats.get("backend_decision"),
            "unsupported_inferrt_input": stats.get("unsupported_inferrt_input"),
        },
        "signature": signature["payload"],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _record_compile_cache_event(
    graph_id: int,
    signature: dict[str, Any],
    stats: dict[str, Any],
) -> None:
    phase = os.environ.get("MS_INFERRT_WARMUP_ACTIVE_PHASE", "").strip()
    bucket = os.environ.get("MS_INFERRT_WARMUP_ACTIVE_BUCKET", "").strip()
    completed = _enabled("MS_INFERRT_WARMUP_COMPLETED", "0")
    signature_hash = str(signature["hash"])

    if phase and bucket:
        _PRECOMPILED_GRAPH_SIGNATURES.add(signature_hash)
        event = "precompile_recorded"
    elif completed:
        if signature_hash in _PRECOMPILED_GRAPH_SIGNATURES:
            event = "compile_invoked_existing_precompile_signature"
        else:
            event = "cache_miss_compile_after_warmup"
            if os.environ.get("MS_INFERRT_CACHE_MISS_POLICY", "").strip().lower() == "error":
                _append_compile_cache_report(graph_id, signature, stats, event, phase, bucket)
                raise RuntimeError(
                    "InferRT graph cache miss after warmup: a new FX graph was "
                    "compiled after MS_INFERRT_WARMUP_COMPLETED=1. "
                    f"signature={signature_hash}."
                )
    else:
        event = "compile_before_extra_warmup"

    _append_compile_cache_report(graph_id, signature, stats, event, phase, bucket)


def apply_optimization_profile(stats: dict[str, Any] | None = None) -> str:
    """Compatibility hook kept for old launch scripts.

    Graph optimization is now delegated to the real ``ms_inferrt`` backend.
    """
    return os.environ.get("MS_INFERRT_EXTERNAL_OPT_PROFILE", "none").strip() or "none"


def optimize_graph(gm: GraphModule) -> dict[str, int]:
    """Apply tiny compatibility fixes before handing the graph to InferRT."""
    changed = 0
    if _enabled("MS_INFERRT_FIX_NONCONTIG_RESHAPE", "1"):
        changed += _insert_contiguous_before_noncontig_reshape(gm)
    return {"changed": changed}


def _shape_tuple(value: Any) -> tuple[str, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    return tuple(str(dim) for dim in shape)


def _insert_contiguous_before_noncontig_reshape(gm: GraphModule) -> int:
    """Make Qwen3.5 reshape semantics explicit for InferRT.

    PyTorch ``reshape`` may materialize a contiguous copy when the input is not
    view-compatible.  InferRT lowers some reshape nodes as pure view ops.  For
    Qwen3.5 linear-attention blocks this can fail at runtime for tensors shaped
    like ``[tokens, heads, head_dim]`` with non-contiguous strides.  Insert an
    explicit ``contiguous`` before the reshape so the graph keeps PyTorch's
    semantics while still compiling the surrounding compute through InferRT.
    """
    changed = 0
    graph = gm.graph
    for node in list(graph.nodes):
        if node.op != "call_method" or node.target not in ("reshape", "view"):
            continue
        if not node.args:
            continue
        source = node.args[0]
        if not hasattr(source, "meta"):
            continue
        source_value = source.meta.get("example_value")
        output_value = node.meta.get("example_value")
        source_shape = _shape_tuple(source_value)
        output_shape = _shape_tuple(output_value)
        if not (
            len(source_shape) == 3
            and len(output_shape) == 2
            and source_shape[-1] == output_shape[-1]
        ):
            continue
        with graph.inserting_before(node):
            contiguous = graph.call_method("contiguous", args=(source,))
        contiguous.meta.update(getattr(source, "meta", {}))
        node.replace_input_with(source, contiguous)
        changed += 1
    if changed:
        graph.lint()
        gm.recompile()
    return changed


def make_backend(original_backend):
    """Wrap ``ms_inferrt.torch.backend`` with compile-cache reporting."""

    def reported_backend(gm: GraphModule, example_inputs):
        graph_id = _next_graph_id()
        opt_stats = optimize_graph(gm)
        stats = _graph_stats(gm, example_inputs)
        stats["external_graph_changes"] = opt_stats.get("changed", 0)
        stats["resolved_profile"] = apply_optimization_profile(stats)
        use_inferrt, backend_reason = _should_use_inferrt_for_graph(stats)
        stats["backend_decision"] = backend_reason
        signature = _graph_signature(gm)
        # FX stores node.kwargs as immutable_dict by design.  That is graph
        # metadata, not a real runtime input to InferRT.  Treating it as an
        # unsupported input incorrectly bypasses InferRT for Qwen3.5 graphs.
        unsupported_input = _contains_unsupported_inferrt_input(example_inputs)
        if unsupported_input:
            stats["unsupported_inferrt_input"] = unsupported_input
        _record_compile_cache_event(graph_id, signature, stats)
        if unsupported_input:
            return gm.forward
        if not use_inferrt:
            return _compile_decode_backend(gm, example_inputs)
        compiled = original_backend(gm, example_inputs)
        return _runtime_dispatch_prefill_only(gm, compiled)

    reported_backend._ms_inferrt_external_backend = True  # type: ignore[attr-defined]
    return reported_backend


__all__ = ["apply_optimization_profile", "make_backend", "optimize_graph"]
