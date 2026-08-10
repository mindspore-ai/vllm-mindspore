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

"""HTTP-level warmup manager for InferRT-backed vLLM services.

This warmup path is intentionally above the worker layer.  It sends synthetic
OpenAI-compatible requests through the same tokenizer, scheduler, chunked
prefill, and torch.compile backend path as real users, then checks InferRT graph
compile reports to see whether a second warmup pass still compiles new graphs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Optional tokenizer integrations are imported lazily and model metadata errors
# are intentionally tolerated so warmup can fall back to explicit limits.
# The functions below are private building blocks of this documented CLI, not a
# public Python API, so repeating the module contract on every helper adds noise.
# pylint: disable=broad-exception-caught,import-outside-toplevel,missing-function-docstring


URL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def parse_lengths(value: str) -> list[int]:
    lengths: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        lengths.append(int(item))
    return sorted(set(length for length in lengths if length > 0))


def parse_ints(value: str) -> list[int]:
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed > 0:
            result.append(parsed)
    return sorted(set(result))


def model_limits(model_path: str | None) -> dict[str, int | None]:
    config_path = Path(model_path or "") / "config.json" if model_path else None
    max_model_len = None
    if config_path is not None and config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            for key in (
                "max_position_embeddings",
                "seq_length",
                "model_max_length",
            ):
                value = data.get(key)
                if isinstance(value, int) and value > 0:
                    max_model_len = value
                    break
        except Exception:
            pass

    for env_name in ("MAX_MODEL_LEN", "MS_INFERRT_WARMUP_MODEL_MAX_LEN"):
        value = os.environ.get(env_name, "").strip()
        if value:
            try:
                max_model_len = int(value)
            except ValueError:
                pass

    max_num_batched_tokens = None
    value = os.environ.get("MAX_NUM_BATCHED_TOKENS", "").strip()
    if value:
        try:
            max_num_batched_tokens = int(value)
        except ValueError:
            pass

    return {
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
    }


def coverage_lengths(
    policy: str,
    explicit_lengths: str,
    model_path: str | None,
    response_margin: int = 0,
) -> list[int]:
    policy = policy.strip().lower()
    if policy in ("", "manual", "custom"):
        return parse_lengths(explicit_lengths)

    limits = model_limits(model_path)
    max_model_len = int(limits.get("max_model_len") or 12288)
    max_num_batched_tokens = int(
        limits.get("max_num_batched_tokens") or min(12288, max_model_len)
    )
    try:
        context_margin = max(
            0,
            int(os.environ.get("MS_INFERRT_WARMUP_CONTEXT_MARGIN", "64")),
        )
    except ValueError:
        context_margin = 64
    context_margin = max(context_margin, response_margin)
    safe_max_model_len = max(1, max_model_len - context_margin)
    upper = max(1, min(safe_max_model_len, max_num_batched_tokens))

    if policy == "smoke":
        candidates = [1, 64, 512, upper]
    elif policy == "prefill":
        candidates = [
            1,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            6144,
            8192,
            10000,
            safe_max_model_len,
        ]
    elif policy == "balanced":
        candidates = [
            1,
            128,
            512,
            1024,
            2048,
            4096,
            8192,
            10000,
            safe_max_model_len,
        ]
    elif policy in ("broad", "broad_lite"):
        candidates = [
            1,
            8,
            16,
            32,
            64,
            128,
            256,
            384,
            512,
            768,
            1024,
            1536,
            2048,
            3072,
            4096,
            6144,
            8192,
            10000,
            safe_max_model_len,
        ]
    else:
        raise ValueError(f"unknown coverage policy: {policy}")

    chunk_edges = [
        max_num_batched_tokens - 1,
        max_num_batched_tokens,
        max_num_batched_tokens + 1,
        max_num_batched_tokens * 2 - 1,
        max_num_batched_tokens * 2,
        max_num_batched_tokens * 2 + 1,
    ]
    candidates.extend(chunk_edges)
    candidates = [min(max(1, item), safe_max_model_len) for item in candidates]
    return sorted(set(candidates))


def representative_prompts(
    prompts: list[dict[str, Any]],
    targets: list[int],
) -> list[dict[str, Any]]:
    if not prompts:
        return []
    selected: dict[int, dict[str, Any]] = {}
    for target in targets:
        item = min(
            prompts,
            key=lambda prompt, target=target: abs(
                int(prompt["target_tokens"]) - target),
        )
        selected[int(item["target_tokens"])] = item
    return list(selected.values())


def build_request_plan(
    policy: str,
    prompts: list[dict[str, Any]],
    decode_max_tokens: list[int],
    batch_sizes: list[int],
) -> list[dict[str, Any]]:
    policy = policy.strip().lower()
    if policy not in ("balanced", "broad_lite"):
        return [
            {
                "prompts": prompts,
                "max_tokens": max_tokens,
                "batch_size": batch_size,
                "reason": "cartesian",
            }
            for max_tokens in decode_max_tokens
            for batch_size in batch_sizes
        ]

    base_max_tokens = decode_max_tokens[0] if decode_max_tokens else 1
    if policy == "broad_lite":
        max_target = max((int(item["target_tokens"]) for item in prompts), default=8192)
        decode_reps = representative_prompts(prompts, [128, 2048, 8192, max_target])
        batch_reps = representative_prompts(prompts, [128, 2048, 8192])
        plan = [
            {
                "prompts": prompts,
                "max_tokens": base_max_tokens,
                "batch_size": 1,
                "reason": "broad_lite_all_prefill_lengths",
            }
        ]
        for max_tokens in decode_max_tokens[1:]:
            plan.append(
                {
                    "prompts": decode_reps,
                    "max_tokens": max_tokens,
                    "batch_size": 1,
                    "reason": "broad_lite_decode_representatives",
                }
            )
        for batch_size in batch_sizes:
            if batch_size <= 1:
                continue
            plan.append(
                {
                    "prompts": batch_reps,
                    "max_tokens": base_max_tokens,
                    "batch_size": batch_size,
                    "reason": "broad_lite_batch_representatives",
                }
            )
        return plan

    plan = [
        {
            "prompts": prompts,
            "max_tokens": base_max_tokens,
            "batch_size": 1,
            "reason": "balanced_all_lengths",
        }
    ]
    reps = representative_prompts(prompts, [128, 2048, 8192])
    for max_tokens in decode_max_tokens[1:]:
        plan.append(
            {
                "prompts": reps,
                "max_tokens": max_tokens,
                "batch_size": 1,
                "reason": "balanced_decode_representatives",
            }
        )
    for batch_size in batch_sizes:
        if batch_size <= 1:
            continue
        plan.append(
            {
                "prompts": reps,
                "max_tokens": base_max_tokens,
                "batch_size": batch_size,
                "reason": "balanced_batch_representatives",
            }
        )
    return plan


def build_verify_plan(
    policy: str,
    prompts: list[dict[str, Any]],
    decode_max_tokens: list[int],
    batch_sizes: list[int],
    verify_policy: str,
) -> list[dict[str, Any]]:
    if verify_policy.strip().lower() != "adaptive":
        return build_request_plan(policy, prompts, decode_max_tokens, batch_sizes)

    policy = policy.strip().lower()
    if policy in ("manual", "smoke"):
        return build_request_plan(policy, prompts, decode_max_tokens, batch_sizes)

    max_target = max((int(item["target_tokens"]) for item in prompts), default=8192)
    all_reps = representative_prompts(prompts, [1, 128, 2048, 8192, max_target])
    path_reps = representative_prompts(prompts, [128, 2048, 8192])
    base_max_tokens = decode_max_tokens[0] if decode_max_tokens else 1
    plan = [
        {
            "prompts": all_reps,
            "max_tokens": base_max_tokens,
            "batch_size": 1,
            "reason": "adaptive_verify_prefill_representatives",
        }
    ]
    for max_tokens in decode_max_tokens[1:]:
        plan.append(
            {
                "prompts": path_reps,
                "max_tokens": max_tokens,
                "batch_size": 1,
                "reason": "adaptive_verify_decode_representatives",
            }
        )
    for batch_size in batch_sizes:
        if batch_size <= 1:
            continue
        plan.append(
            {
                "prompts": path_reps,
                "max_tokens": base_max_tokens,
                "batch_size": batch_size,
                "reason": "adaptive_verify_batch_representatives",
            }
        )
    return plan


def wait_health(base_url: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    url = f"{base_url.rstrip('/')}/health"
    while time.time() < deadline:
        try:
            with URL_OPENER.open(url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"service did not become healthy: {url}")


def load_tokenizer(model_path: str | None) -> Any | None:
    if not model_path:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=os.environ.get(
                "MS_INFERRT_SERVICE_WARMUP_TRUST_REMOTE_CODE", "0"
            )
            not in ("", "0", "false", "False", "FALSE"),
        )
    except Exception:
        return None


def token_len(tokenizer: Any | None, text: str, use_chat_template: bool) -> int:
    if tokenizer is None:
        return len(text)
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return len(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def build_prompt(target_tokens: int, tokenizer: Any | None, use_chat_template: bool) -> str:
    marker = f"[INFERRT_WARMUP_TARGET_{target_tokens}] "
    unit = (
        "这是一段用于服务启动预热的合成文本，只用于触发真实请求路径、"
        "调度路径、chunked prefill 路径和 InferRT 编译路径，不代表用户数据。"
    )
    prefix = "/no_think\n"
    base = marker + unit
    base_tokens = token_len(tokenizer, prefix + base, use_chat_template)
    if base_tokens >= target_tokens:
        text = base
    else:
        unit_tokens = max(1, token_len(tokenizer, "\n" + unit, use_chat_template))
        repeat_count = max(1, (target_tokens - base_tokens + unit_tokens - 1) // unit_tokens)
        text = base + ("\n" + unit) * repeat_count
        while token_len(tokenizer, prefix + text, use_chat_template) < target_tokens:
            repeat_count = max(1, repeat_count)
            text += ("\n" + unit) * repeat_count

    # Trim by characters to avoid huge overshoot while keeping deterministic text.
    lo, hi = 1, len(text)
    best = text
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid]
        if token_len(tokenizer, prefix + candidate, use_chat_template) <= target_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def send_chat_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"/no_think\n{prompt}"}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with URL_OPENER.open(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    elapsed_s = time.perf_counter() - started
    return {
        "elapsed_s": elapsed_s,
        "usage": data.get("usage"),
        "finish_reason": (
            (data.get("choices") or [{}])[0].get("finish_reason")
            if isinstance(data, dict)
            else None
        ),
    }


def send_request_group(
    base_url: str,
    model: str,
    prompts: list[dict[str, Any]],
    max_tokens: int,
    timeout: int,
    batch_size: int,
    single_request_parallelism: int,
) -> list[dict[str, Any]]:
    if batch_size <= 1:
        results = []
        workers = max(1, single_request_parallelism)
        for start in range(0, len(prompts), workers):
            group = prompts[start : start + workers]
            if len(group) == 1:
                item = group[0]
                result = send_chat_request(
                    base_url,
                    model,
                    str(item["prompt"]),
                    max_tokens,
                    timeout,
                )
                results.append(
                    {
                        "target_tokens": item["target_tokens"],
                        "actual_tokens": item["actual_tokens"],
                        "batch_size": 1,
                        "http_concurrency": 1,
                        "max_tokens": max_tokens,
                        **result,
                    }
                )
                continue
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(group)) as executor:
                future_to_item = {
                    executor.submit(
                        send_chat_request,
                        base_url,
                        model,
                        str(item["prompt"]),
                        max_tokens,
                        timeout,
                    ): item
                    for item in group
                }
                for future in concurrent.futures.as_completed(future_to_item):
                    item = future_to_item[future]
                    result = future.result()
                    results.append(
                        {
                            "target_tokens": item["target_tokens"],
                            "actual_tokens": item["actual_tokens"],
                            "batch_size": 1,
                            "http_concurrency": len(group),
                            "max_tokens": max_tokens,
                            **result,
                        }
                    )
        return results

    results: list[dict[str, Any]] = []
    for start in range(0, len(prompts), batch_size):
        group = prompts[start : start + batch_size]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(group)) as executor:
            future_to_item = {
                executor.submit(
                    send_chat_request,
                    base_url,
                    model,
                    str(item["prompt"]),
                    max_tokens,
                    timeout,
                ): item
                for item in group
            }
            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                result = future.result()
                results.append(
                    {
                        "target_tokens": item["target_tokens"],
                        "actual_tokens": item["actual_tokens"],
                        "batch_size": len(group),
                        "http_concurrency": len(group),
                        "max_tokens": max_tokens,
                        **result,
                    }
                )
    return results


def read_compile_reports(report_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(report_dir.glob("graph_compile_cache_pid*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_path"] = str(path)
                rows.append(row)
    return rows


def summarize_compile_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    events: dict[str, int] = {}
    hashes: set[str] = set()
    for row in rows:
        events[row.get("cache_event", "unknown")] = (
            events.get(row.get("cache_event", "unknown"), 0) + 1
        )
        if row.get("signature_hash"):
            hashes.add(str(row["signature_hash"]))
    return {
        "compile_event_count": len(rows),
        "unique_signature_count": len(hashes),
        "events": events,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8162")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--lengths", default="128,512,1024,2048,4096,8192,10000")
    parser.add_argument(
        "--coverage-policy",
        default="manual",
        choices=("manual", "smoke", "prefill", "balanced", "broad_lite", "broad"),
        help=(
            "Generate model-aware fake requests that cover likely compile "
            "paths. 'manual' uses --lengths exactly."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument(
        "--decode-max-tokens",
        default="",
        help=(
            "Comma-separated max_tokens values to cover decode paths. If set, "
            "this overrides --max-tokens for warmup."
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        default="1",
        help=(
            "Comma-separated concurrent HTTP request group sizes used to "
            "exercise scheduler batching paths."
        ),
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--health-timeout-s", type=int, default=300)
    parser.add_argument("--request-timeout-s", type=int, default=900)
    parser.add_argument(
        "--single-request-parallelism",
        type=int,
        default=1,
        help=(
            "Optional parallelism for warmup plan steps whose logical batch_size "
            "is 1. Keep 1 for the most conservative startup; use 2/4 to reduce "
            "broad warmup wall time when scheduler batching side effects are OK."
        ),
    )
    parser.add_argument(
        "--verify-policy",
        default="full",
        choices=("full", "adaptive"),
        help=(
            "Warmup verification plan for rounds after the first. 'full' repeats "
            "the same request plan every round. 'adaptive' runs the first round "
            "with full coverage, then validates only short/mid/long and boundary "
            "representatives to reduce startup wall time."
        ),
    )
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--ready-file", default="")
    parser.add_argument("--use-chat-template", action="store_true")
    return parser.parse_args()


def _plan_summary(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "reason": step["reason"],
        "prompt_count": len(step["prompts"]),
        "target_tokens": [int(item["target_tokens"])
                          for item in step["prompts"]],
        "max_tokens": step["max_tokens"],
        "batch_size": step["batch_size"],
    } for step in plan]


def _prepare_plans(args: argparse.Namespace
                   ) -> tuple[Path, list[int], list[int],
                              list[dict[str, Any]], list[dict[str, Any]]]:
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.model_path or args.model)
    decode_tokens = (parse_ints(args.decode_max_tokens)
                     if args.decode_max_tokens.strip() else [args.max_tokens])
    lengths = coverage_lengths(
        args.coverage_policy,
        args.lengths,
        args.model_path or args.model,
        response_margin=(max(decode_tokens) + 64 if decode_tokens else 64),
    )
    batch_sizes = parse_ints(args.batch_sizes) or [1]
    prompts = [{
        "target_tokens": length,
        "prompt": build_prompt(length, tokenizer, args.use_chat_template),
    } for length in lengths]
    for item in prompts:
        item["actual_tokens"] = token_len(
            tokenizer, f"/no_think\n{item['prompt']}", args.use_chat_template)
    request_plan = build_request_plan(
        args.coverage_policy, prompts, decode_tokens, batch_sizes)
    verify_plan = build_verify_plan(
        args.coverage_policy, prompts, decode_tokens, batch_sizes,
        args.verify_policy)
    return report_dir, lengths, batch_sizes, request_plan, verify_plan


def _run_round(args: argparse.Namespace, plan: list[dict[str, Any]],
               report_dir: Path, round_idx: int,
               previous_compile_count: int) -> tuple[dict[str, Any], int]:
    request_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in plan:
        request_results.extend(send_request_group(
            args.base_url,
            args.model,
            list(step["prompts"]),
            int(step["max_tokens"]),
            args.request_timeout_s,
            int(step["batch_size"]),
            args.single_request_parallelism,
        ))
    rows = read_compile_reports(report_dir)
    new_rows = rows[previous_compile_count:]
    result = {
        "round": round_idx,
        "elapsed_s": time.perf_counter() - started,
        "request_count": len(request_results),
        "avg_request_elapsed_s": (statistics.mean(
            item["elapsed_s"] for item in request_results)
                                  if request_results else None),
        "requests": request_results,
        "new_compile_event_count": len(new_rows),
        "new_compile_events": summarize_compile_rows(new_rows),
        "total_compile_events": summarize_compile_rows(rows),
        "request_plan": _plan_summary(plan),
    }
    return result, len(rows)


def _write_result(args: argparse.Namespace, payload: dict[str, Any],
                  converged: bool) -> None:
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    if args.ready_file and converged:
        ready = Path(args.ready_file)
        ready.parent.mkdir(parents=True, exist_ok=True)
        ready.write_text(json.dumps({"ready": True, "json": str(out)}) + "\n",
                         encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> int:
    args = _parse_args()
    report_dir, lengths, batch_sizes, request_plan, verify_plan = _prepare_plans(args)

    decode_max_tokens = (
        parse_ints(args.decode_max_tokens)
        if args.decode_max_tokens.strip()
        else [args.max_tokens]
    )
    wait_health(args.base_url, args.health_timeout_s)
    rounds: list[dict[str, Any]] = []
    previous_compile_count = len(read_compile_reports(report_dir))
    for round_idx in range(1, args.rounds + 1):
        active_plan = request_plan if round_idx == 1 else verify_plan
        result, previous_compile_count = _run_round(
            args, active_plan, report_dir, round_idx, previous_compile_count)
        rounds.append(result)

    converged = bool(rounds) and rounds[-1]["new_compile_event_count"] == 0
    payload = {
        "base_url": args.base_url,
        "model": args.model,
        "model_path": args.model_path,
        "lengths": lengths,
        "coverage_policy": args.coverage_policy,
        "max_tokens": args.max_tokens,
        "decode_max_tokens": decode_max_tokens,
        "batch_sizes": batch_sizes,
        "single_request_parallelism": args.single_request_parallelism,
        "verify_policy": args.verify_policy,
        "request_plan": _plan_summary(request_plan),
        "verify_plan": _plan_summary(verify_plan),
        "rounds": rounds,
        "converged": converged,
        "meaning": (
            "converged=True means the final warmup round did not trigger any "
            "new backend compile events in graph_compile_cache reports."
        ),
    }
    _write_result(args, payload, converged)
    return 0 if converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
