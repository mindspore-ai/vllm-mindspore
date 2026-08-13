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

"""Qwen3.5 compatibility patches for the direct InferRT backend.

These are compatibility fixes rather than optional performance experiments.
They avoid FX nodes that the current InferRT backend cannot lower, while
keeping the outer vLLM torch.compile backend set to InferRT.
"""

# Optional version-dependent modules are imported lazily. Compatibility
# failures intentionally fall back to the unpatched implementation.
# pylint: disable=import-outside-toplevel,broad-exception-caught,protected-access

from __future__ import annotations

import os
from typing import Any


def _env_enabled(name: str, default: str = "1") -> bool:
    value = os.environ.get(name, default)
    return value not in ("", "0", "false", "False", "FALSE")


def _enabled() -> bool:
    return os.environ.get("VLLM_TORCH_COMPILE_BACKEND", "").lower() == "inferrt"


def _log(message: str, *args: Any) -> None:
    try:
        from vllm.logger import init_logger

        init_logger("inferrt_vllm_external_patch").info(message, *args)
    except Exception:
        print(message % args if args else message)


def patch_qwen35_gdn_capture_guard() -> None:
    """Graph-break Qwen3.5 GDN kernels that carry immutable_dict kwargs.

    vLLM's Qwen3.5 prefill path can emit higher-order Triton wrapper nodes.
    Their ``immutable_dict`` kwargs are metadata for Torch execution, not
    InferRT IR values.  Disabling compile for the GDN implementation prevents
    those wrapper nodes from entering the InferRT graph.
    """

    if not _enabled() or not _env_enabled("MS_INFERRT_QWEN35_GDN_CAPTURE_GUARD", "1"):
        return

    try:
        import torch
        from vllm.model_executor.models import qwen3_next
    except Exception as exc:
        _log("Skip Qwen3.5 GDN capture guard: %s", exc)
        return

    cls = getattr(qwen3_next, "ChunkGatedDeltaRule", None)
    if cls is None or getattr(cls, "_inferrt_gdn_capture_guard", False):
        return

    def disable_compile(fn: Any) -> Any:
        try:
            return torch.compiler.disable(fn, recursive=True)
        except Exception:
            try:
                import torch._dynamo as dynamo

                return dynamo.disable(fn, recursive=True)
            except Exception:
                return fn

    patched = []
    for name in ("forward_native", "forward_cuda"):
        fn = getattr(cls, name, None)
        if fn is None:
            continue
        setattr(cls, name, disable_compile(fn))
        patched.append(name)

    cls._inferrt_gdn_capture_guard = True
    if patched:
        _log(
            "Enabled Qwen3.5 GDN capture guard for %s. GDN kernels graph-break "
            "while surrounding graphs use InferRT.",
            ",".join(patched),
        )


def patch_qwen35_gated_rmsnorm_native() -> None:
    """Use native gated RMSNorm to avoid Triton HOP capture in Qwen3.5."""

    if not _enabled() or not _env_enabled("MS_INFERRT_QWEN35_NATIVE_GATED_RMS", "1"):
        return

    try:
        from vllm.model_executor.layers.layernorm import RMSNormGated
        from vllm_ascend.ops.layernorm import AscendRMSNormGated
    except Exception as exc:
        _log("Skip Qwen3.5 gated RMSNorm native patch: %s", exc)
        return

    if getattr(AscendRMSNormGated, "_inferrt_native_gated_rms", False):
        return

    AscendRMSNormGated.forward_oot = RMSNormGated.forward_native
    AscendRMSNormGated._inferrt_native_gated_rms = True
    _log(
        "Enabled Qwen3.5 native gated RMSNorm for InferRT. Triton gated "
        "layernorm wrappers will not be captured."
    )


def patch_qwen35_full_graph_fia_graph_break() -> None:
    """Keep vLLM-Ascend full_graph_fia out of InferRT python_call lowering.

    Qwen3.5 on vLLM-Ascend can call ``AscendAttentionBackendImpl.full_graph_fia``
    while vLLM profiles/prefills the model.  The function reads
    ``get_graph_params().workspaces`` from vLLM-Ascend's runtime context.  When
    Dynamo captures it into an InferRT python_call, InferRT asks the Python call
    to calculate workspace outside that context and ``graph_params`` can be
    ``None``.  Graph-breaking only this method keeps the outer prefill graph on
    the direct InferRT backend while leaving the vLLM-Ascend attention function
    on its original torch_npu path.
    """

    if not _enabled() or not _env_enabled("MS_INFERRT_QWEN35_FIA_GRAPH_BREAK", "1"):
        return

    try:
        import torch
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
    except Exception as exc:
        _log("Skip Qwen3.5 full_graph_fia graph break: %s", exc)
        return

    if getattr(AscendAttentionBackendImpl, "_inferrt_fia_graph_break", False):
        return

    fn = getattr(AscendAttentionBackendImpl, "full_graph_fia", None)
    if fn is None:
        return

    try:
        patched = torch.compiler.disable(fn, recursive=True)
    except Exception:
        try:
            import torch._dynamo as dynamo

            patched = dynamo.disable(fn, recursive=True)
        except Exception:
            patched = fn

    AscendAttentionBackendImpl.full_graph_fia = patched
    AscendAttentionBackendImpl._inferrt_fia_graph_break = True
    _log(
        "Enabled Qwen3.5 full_graph_fia graph break for InferRT. "
        "Outer prefill graphs still use InferRT; full_graph_fia keeps the "
        "original vLLM-Ascend torch_npu path."
    )


def apply_qwen35_compat_patches() -> None:
    patch_qwen35_gdn_capture_guard()
    patch_qwen35_gated_rmsnorm_native()
    patch_qwen35_full_graph_fia_graph_break()


__all__ = ["apply_qwen35_compat_patches"]
