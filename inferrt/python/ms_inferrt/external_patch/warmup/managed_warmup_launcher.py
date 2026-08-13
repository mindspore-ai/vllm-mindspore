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

"""Managed InferRT vLLM launcher with HTTP-level warmup gating.

This process starts the normal external InferRT vLLM launcher as a child
process, waits until the service is healthy, runs service_warmup_manager.py, and
only then writes a ready file.  Production traffic should be gated on the ready
file or equivalent readiness probe.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# This launcher must keep a long-lived Popen and deliberately converts startup
# failures into warmup reports before terminating the complete process group.
# pylint: disable=broad-exception-caught,consider-using-with,missing-function-docstring


PATCH_DIR = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
WARMUP_ROOT = PATCH_DIR
BACKEND_PATCH_ROOT = PATCH_DIR.parent / "register"
SERVER_LAUNCHER = BACKEND_PATCH_ROOT / "inferrt_vllm_backend_launcher.py"
WARMUP_MANAGER = PATCH_DIR / "service_warmup_manager.py"


def _parse_server_value(server_args: list[str], name: str, default: str) -> str:
    flag = f"--{name}"
    for idx, item in enumerate(server_args):
        if item == flag and idx + 1 < len(server_args):
            return server_args[idx + 1]
        if item.startswith(flag + "="):
            return item.split("=", 1)[1]
    return default


def _parse_model(server_args: list[str]) -> str:
    if "serve" in server_args:
        idx = server_args.index("serve")
        if idx + 1 < len(server_args):
            return server_args[idx + 1]
    for item in server_args:
        if not item.startswith("-"):
            return item
    raise SystemExit("Could not infer model path from vLLM server arguments.")


def _parse_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _auto_warmup_lengths(server_args: list[str]) -> str:
    max_model_len = _parse_int(_parse_server_value(server_args, "max-model-len", "12288"), 12288)
    max_num_batched_tokens = _parse_int(
        _parse_server_value(server_args, "max-num-batched-tokens", "4096"),
        4096,
    )
    upper = max(1, min(max_model_len, max_num_batched_tokens))
    candidates = {128, 512, 1024, 2048, 4096, upper}
    if max_model_len > upper:
        # Include long user prompts so the HTTP warmup also validates chunked
        # prefill and scheduler paths, even though one executable chunk is
        # capped by max_num_batched_tokens.
        candidates.update({min(max_model_len, upper * 2), max_model_len})
    lengths = sorted({item for item in candidates if 0 < item <= max_model_len})
    return ",".join(str(item) for item in lengths)


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)


def _run_warmup(
    base_url: str,
    model: str,
    report_dir: Path,
    ready_file: Path,
    json_out: Path,
    lengths: str,
    coverage_policy: str,
    decode_max_tokens: str,
    batch_sizes: str,
    rounds: int,
    max_tokens: int,
    timeout_s: int,
    single_request_parallelism: int,
    verify_policy: str,
    env: dict[str, str],
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            str(PYTHON),
            str(WARMUP_MANAGER),
            "--base-url",
            base_url,
            "--model",
            model,
            "--model-path",
            model,
            "--report-dir",
            str(report_dir),
            "--lengths",
            lengths,
            "--coverage-policy",
            coverage_policy,
            "--max-tokens",
            str(max_tokens),
            "--decode-max-tokens",
            decode_max_tokens,
            "--batch-sizes",
            batch_sizes,
            "--rounds",
            str(rounds),
            "--health-timeout-s",
            str(timeout_s),
            "--single-request-parallelism",
            str(single_request_parallelism),
            "--verify-policy",
            verify_policy,
            "--json-out",
            str(json_out),
            "--ready-file",
            str(ready_file),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--warmup-json", default="")
    parser.add_argument(
        "--warmup-lengths",
        default="auto",
        help=(
            "HTTP-level warmup lengths. Use 'auto' to derive model-aware "
            "lengths from --max-model-len and --max-num-batched-tokens."
        ),
    )
    parser.add_argument(
        "--coverage-policy",
        default="broad",
        choices=("manual", "smoke", "prefill", "balanced", "broad_lite", "broad"),
        help=(
            "HTTP fake-request coverage policy before readiness. 'broad' "
            "covers short/mid/long prefill, chunk boundaries, decode variants, "
            "and optional scheduler batching."
        ),
    )
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--warmup-max-tokens", type=int, default=1)
    parser.add_argument("--warmup-decode-max-tokens", default="1")
    parser.add_argument("--warmup-batch-sizes", default="1")
    parser.add_argument("--warmup-single-request-parallelism", type=int, default=1)
    parser.add_argument(
        "--warmup-verify-policy",
        default="full",
        choices=("full", "adaptive"),
    )
    parser.add_argument(
        "--worker-warmup-source",
        default="auto",
        choices=("auto", "prompts", "profile_adaptive", "manual", "none"),
        help=(
            "Worker-level precompile strategy before readiness. 'auto' uses "
            "prompts when --worker-warmup-prompts-file is set, otherwise "
            "profile_adaptive."
        ),
    )
    parser.add_argument("--worker-warmup-prompts-file", default="")
    parser.add_argument("--worker-warmup-max-buckets", type=int, default=6)
    parser.add_argument("--worker-warmup-bucket-granularity", type=int, default=128)
    parser.add_argument("--worker-warmup-profile-candidates", default="")
    parser.add_argument("--worker-warmup-profile-max-candidates", type=int, default=12)
    parser.add_argument(
        "--cache-miss-policy",
        default="",
        choices=("", "report", "error"),
        help=(
            "What to do if a real request compiles a graph after worker "
            "warmup. Empty/report only records the miss; error raises inside "
            "the backend and is useful for coverage validation."
        ),
    )
    parser.add_argument("--health-timeout-s", type=int, default=300)
    parser.add_argument("--server-log", default="")
    parser.add_argument("--allow-warmup-failure", action="store_true")
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _prepare_context(args: argparse.Namespace) -> dict[str, Any]:
    server_args = list(args.server_args)
    if server_args and server_args[0] == "--":
        server_args = server_args[1:]
    if not server_args:
        raise SystemExit("Pass vLLM serve arguments after '--'.")

    ready_file = Path(args.ready_file)
    ready_file.unlink(missing_ok=True)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    warmup_json = (Path(args.warmup_json) if args.warmup_json else
                   report_dir / "service_warmup_report.json")
    host = _parse_server_value(server_args, "host", "127.0.0.1")
    port = _parse_server_value(server_args, "port", "8000")
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    model = _parse_model(server_args)
    max_model_len = _parse_server_value(server_args, "max-model-len", "12288")
    max_batched = _parse_server_value(
        server_args, "max-num-batched-tokens", "4096")
    lengths = (_auto_warmup_lengths(server_args)
               if args.warmup_lengths.strip().lower() == "auto"
               else args.warmup_lengths)
    return {
        "server_args": server_args,
        "ready_file": ready_file,
        "report_dir": report_dir,
        "warmup_json": warmup_json,
        "base_url": f"http://{probe_host}:{port}",
        "model": model,
        "max_model_len": max_model_len,
        "max_batched": max_batched,
        "lengths": lengths,
    }


def _server_environment(args: argparse.Namespace, context: dict[str, Any]
                        ) -> tuple[dict[str, str], str]:
    report_dir = context["report_dir"]
    model = context["model"]
    max_model_len = context["max_model_len"]
    env = os.environ.copy()
    env.setdefault("MS_INFERRT_WARMUP_ROOT", str(WARMUP_ROOT))
    env.setdefault("VLLM_TORCH_COMPILE_BACKEND", "inferrt")
    env.setdefault("MS_INFERRT_EXTERNAL_OPT_PROFILE", "prefill")
    env.setdefault("MS_INFERRT_GRAPH_CACHE_REPORT", "1")
    env.setdefault("MS_INFERRT_DEV_DUMP_IR", "1")
    env.setdefault("NO_PROXY", "127.0.0.1,localhost")
    env.setdefault("no_proxy", "127.0.0.1,localhost")
    env.setdefault("MS_INFERRT_INTERNAL_REPORT_DIR", str(report_dir))
    env.setdefault("MS_INFERRT_INTERNAL_REPORT", "0")
    env.setdefault("MS_INFERRT_ATTENTION_METADATA_REPORT", "0")

    worker_source = args.worker_warmup_source
    if worker_source == "auto":
        worker_source = ("prompts" if args.worker_warmup_prompts_file
                         else "profile_adaptive")
    if worker_source == "none":
        env["MS_INFERRT_PATCH_WARMUP"] = "0"
    else:
        env.update({
            "MS_INFERRT_PATCH_WARMUP": "1",
            "MS_INFERRT_WARMUP_PREFILL_SOURCE": worker_source,
            "MS_INFERRT_WARMUP_MODEL_PATH": model,
            "MS_INFERRT_WARMUP_MODEL_MAX_LEN": max_model_len,
            "MS_INFERRT_WARMUP_REPORT_DIR": str(report_dir),
            "MS_INFERRT_WARMUP_MAX_BUCKETS": str(args.worker_warmup_max_buckets),
            "MS_INFERRT_WARMUP_BUCKET_GRANULARITY": str(
                args.worker_warmup_bucket_granularity),
            "MS_INFERRT_WARMUP_PROFILE_MAX_CANDIDATES": str(
                args.worker_warmup_profile_max_candidates),
        })
        if args.worker_warmup_prompts_file:
            env["MS_INFERRT_WARMUP_PROMPTS_FILE"] = args.worker_warmup_prompts_file
        if args.worker_warmup_profile_candidates:
            env["MS_INFERRT_WARMUP_PROFILE_CANDIDATES"] = (
                args.worker_warmup_profile_candidates)
    if args.cache_miss_policy == "error":
        env["MS_INFERRT_CACHE_MISS_POLICY"] = "error"
    return env, worker_source


def _start_server(args: argparse.Namespace, server_args: list[str],
                  env: dict[str, str]) -> tuple[subprocess.Popen, Any]:
    log_handle = None
    stdout = None
    if args.server_log:
        log_path = Path(args.server_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        stdout = log_handle
    proc = subprocess.Popen(
        [str(PYTHON), str(SERVER_LAUNCHER), *server_args],
        env=env,
        stdout=stdout,
        stderr=subprocess.STDOUT if stdout is not None else None,
        text=True,
        start_new_session=True,
    )
    return proc, log_handle


def main() -> int:
    args = _parse_args()
    context = _prepare_context(args)

    env, worker_source = _server_environment(args, context)
    server_args = context["server_args"]
    ready_file = context["ready_file"]
    report_dir = context["report_dir"]
    warmup_json = context["warmup_json"]
    base_url = context["base_url"]
    model = context["model"]
    max_model_len = context["max_model_len"]
    max_num_batched_tokens = context["max_batched"]
    warmup_lengths = context["lengths"]

    warmup_env = os.environ.copy()
    warmup_env["MAX_MODEL_LEN"] = max_model_len
    warmup_env["MAX_NUM_BATCHED_TOKENS"] = max_num_batched_tokens

    proc, log_handle = _start_server(args, server_args, env)

    def handle_signal(signum: int, frame: Any) -> None:
        del frame
        _stop_process(proc)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        warmup = _run_warmup(
            base_url,
            model,
            report_dir,
            ready_file,
            warmup_json,
            warmup_lengths,
            args.coverage_policy,
            args.warmup_decode_max_tokens,
            args.warmup_batch_sizes,
            args.warmup_rounds,
            args.warmup_max_tokens,
            args.health_timeout_s,
            args.warmup_single_request_parallelism,
            args.warmup_verify_policy,
            warmup_env,
        )
        (report_dir / "service_warmup_manager.out").write_text(
            warmup.stdout,
            encoding="utf-8",
        )
        if warmup.returncode != 0 and not args.allow_warmup_failure:
            _stop_process(proc)
            print(warmup.stdout[-4000:], file=sys.stderr)
            return warmup.returncode

        if warmup.returncode == 0:
            print(
                json.dumps(
                    {
                        "ready": True,
                        "ready_file": str(ready_file),
                        "warmup_json": str(warmup_json),
                        "base_url": base_url,
                        "worker_warmup_source": worker_source,
                        "coverage_policy": args.coverage_policy,
                        "warmup_lengths": warmup_lengths,
                        "warmup_decode_max_tokens": args.warmup_decode_max_tokens,
                        "warmup_batch_sizes": args.warmup_batch_sizes,
                        "warmup_single_request_parallelism": args.warmup_single_request_parallelism,
                        "warmup_verify_policy": args.warmup_verify_policy,
                        "max_model_len": max_model_len,
                        "max_num_batched_tokens": max_num_batched_tokens,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        else:
            print(
                json.dumps(
                    {
                        "ready": False,
                        "warmup_failed_but_allowed": True,
                        "warmup_json": str(warmup_json),
                        "base_url": base_url,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        while proc.poll() is None:
            time.sleep(2)
        return int(proc.returncode or 0)
    finally:
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
