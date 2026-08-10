#!/usr/bin/env python3
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

"""Launch vLLM with InferRT as a torch.compile backend without editing vLLM.

This file is intentionally standalone. It monkey-patches vLLM in the current
Python process before vLLM's CLI creates the engine, so users do not need to
modify vLLM or vLLM-Ascend source files.

Example:
    python inferrt/python/ms_inferrt/external_patch/register/\
inferrt_vllm_backend_launcher.py serve \
      /path/to/Qwen3-8B \
      --compilation-config '{"backend":"inferrt"}'
"""

# Imports are intentionally delayed until environment setup and patch
# installation have completed.
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import os
import sys
from pathlib import Path


PATCH_DIR = Path(__file__).resolve().parent
WARMUP_PATCH_DIR = PATCH_DIR.parent / "warmup"


def apply_patch() -> None:
    """Install the external InferRT backend patch for this process and workers."""
    os.environ.setdefault("VLLM_TORCH_COMPILE_BACKEND", "inferrt")
    os.environ.setdefault("VLLM_USE_BYTECODE_HOOK", "0")
    os.environ.setdefault("MS_INFERRT_EXTERNAL_OPT_PROFILE", "prefill")
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    os.environ["INFERRT_VLLM_EXTERNAL_PATCH"] = "1"

    patch_dir_text = str(PATCH_DIR)
    if patch_dir_text not in sys.path:
        sys.path.insert(0, patch_dir_text)
    warmup_patch_dir_text = str(WARMUP_PATCH_DIR)
    if WARMUP_PATCH_DIR.is_dir() and warmup_patch_dir_text not in sys.path:
        sys.path.insert(0, warmup_patch_dir_text)

    pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [p for p in pythonpath.split(os.pathsep) if p]
    if patch_dir_text not in pythonpath_parts:
        pythonpath_parts.insert(0, patch_dir_text)
    if (
        WARMUP_PATCH_DIR.is_dir()
        and warmup_patch_dir_text not in pythonpath_parts
    ):
        pythonpath_parts.insert(0, warmup_patch_dir_text)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    from inferrt_patch import apply_patch as external_apply_patch

    external_apply_patch()


def main() -> None:
    apply_patch()
    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
