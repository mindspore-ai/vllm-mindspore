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

"""Extra InferRT warmup hooks for vLLM-Ascend.

The patch is opt-in and intentionally conservative.  It extends vLLM-Ascend's
normal compile/warmup phase with additional dummy prefill runs so InferRT can
see more token-count buckets before serving real requests.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

# This optional compatibility plugin deliberately catches third-party import
# and runtime failures so a missing vLLM-Ascend feature does not break startup.
# Its private helpers implement the module contract and are not a public API.
# pylint: disable=broad-exception-caught,import-outside-toplevel,protected-access,missing-function-docstring


PATCH_DIR = Path(__file__).resolve().parent
WARMUP_ROOT = PATCH_DIR
RUNTIME_DIR = WARMUP_ROOT / "runtime"


def _env_enabled(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    return value not in ("", "0", "false", "False", "FALSE")


def _log(message: str, *args: Any) -> None:
    try:
        from vllm.logger import init_logger

        init_logger("inferrt_vllm_external_patch").info(message, *args)
    except Exception:
        print(message % args if args else message)


def _parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parsed = int(item)
        except ValueError:
            _log("Ignore invalid InferRT warmup bucket %r.", item)
            continue
        if parsed > 0:
            result.append(parsed)
    return result


def _parse_prefill_ranges(value: str) -> tuple[list[int], list[dict[str, int | str]]]:
    buckets: list[int] = []
    ranges: list[dict[str, int | str]] = []
    policy = os.environ.get(
        "MS_INFERRT_WARMUP_RANGE_POLICY", "upper"
    ).strip().lower()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            try:
                point = int(item)
            except ValueError:
                _log("Ignore invalid InferRT warmup range %r.", item)
                continue
            if point <= 0:
                continue
            buckets.append(point)
            ranges.append({"range": item, "min": point, "max": point, "bucket": point})
            continue

        left, right = item.split("-", 1)
        try:
            lo = int(left.strip())
            hi = int(right.strip())
        except ValueError:
            _log("Ignore invalid InferRT warmup range %r.", item)
            continue
        if lo <= 0 or hi <= 0 or lo > hi:
            _log("Ignore invalid InferRT warmup range %r.", item)
            continue

        if policy == "middle":
            bucket = (lo + hi) // 2
        else:
            bucket = hi
        buckets.append(bucket)
        ranges.append({"range": item, "min": lo, "max": hi, "bucket": bucket})
    return buckets, ranges


def _unique_sorted_desc(values: list[int]) -> list[int]:
    return sorted(set(values), reverse=True)


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return int(math.ceil(value / multiple) * multiple)


def _max_dummy_tokens(model_runner: Any) -> int | None:
    for attr in ("max_num_tokens",):
        value = getattr(model_runner, attr, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass

    scheduler_config = getattr(model_runner, "scheduler_config", None)
    value = getattr(scheduler_config, "max_num_batched_tokens", None)
    if value is not None:
        try:
            return int(value)
        except Exception:
            pass
    return None


def _model_max_len(model_runner: Any) -> int | None:
    for env_name in (
        "MS_INFERRT_WARMUP_MODEL_MAX_LEN",
        "MAX_MODEL_LEN",
    ):
        value = os.environ.get(env_name, "").strip()
        if value:
            try:
                return int(value)
            except ValueError:
                _log("Ignore invalid model max length from %s=%r.", env_name, value)

    vllm_config = getattr(model_runner, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    for attr in ("max_model_len", "max_seq_len"):
        value = getattr(model_config, attr, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
    return None


def _model_path(model_runner: Any) -> str | None:
    for env_name in (
        "MS_INFERRT_WARMUP_MODEL_PATH",
        "PREFILL_MODEL",
        "MODEL_PATH",
    ):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value

    vllm_config = getattr(model_runner, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    for attr in ("model", "tokenizer"):
        value = getattr(model_config, attr, None)
        if value:
            return str(value)
    return None


def _prompt_file() -> Path | None:
    for env_name in (
        "MS_INFERRT_WARMUP_PROMPTS_FILE",
        "PREFILL_PROMPTS",
    ):
        value = os.environ.get(env_name, "").strip()
        if value:
            return Path(value)
    return None


def _prompt_text(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("prompt", "text", "content"):
            value = item.get(key)
            if isinstance(value, str):
                return value
    return None


def _append_prompt_items(texts: list[str], items: Any, limit: int) -> None:
    if not isinstance(items, list):
        return
    for item in items:
        text = _prompt_text(item)
        if text is not None:
            texts.append(text)
        if len(texts) >= limit:
            return


def _read_line_prompts(path: Path, limit: int) -> list[str]:
    texts: list[str] = []
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.rstrip("\n")
            if not line:
                continue
            item: Any = line
            if suffix == ".jsonl":
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    pass
            text = _prompt_text(item)
            if text is not None:
                texts.append(text)
            if len(texts) >= limit:
                break
    return texts


def _read_prompt_texts(path: Path, limit: int) -> list[str]:
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            items = data
            if isinstance(data, dict):
                items = data.get("prompts") or data.get("data") or data.get("results")
            texts: list[str] = []
            _append_prompt_items(texts, items, limit)
            return texts
        return _read_line_prompts(path, limit)
    except Exception as exc:
        _log("Failed to read InferRT warmup prompt file %s: %r", path, exc)
        return []


def _token_lengths_for_prompts(model_runner: Any) -> list[int]:
    path = _prompt_file()
    if path is None or not path.exists():
        _log("Skip dynamic warmup prompts: prompt file does not exist: %s", path)
        return []

    try:
        limit = max(1, int(os.environ.get("MS_INFERRT_WARMUP_PROMPTS_LIMIT", "1000")))
    except ValueError:
        limit = 1000
    texts = _read_prompt_texts(path, limit)
    if not texts:
        _log("Skip dynamic warmup prompts: no prompts loaded from %s.", path)
        return []

    model_path = _model_path(model_runner)
    if not model_path:
        _log("Skip dynamic warmup prompts: model path was not found.")
        return []

    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        _log("Skip dynamic warmup prompts: transformers import failed: %r", exc)
        return []

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=os.environ.get(
                "MS_INFERRT_WARMUP_TRUST_REMOTE_CODE", "0"
            )
            not in ("", "0", "false", "False", "FALSE"),
        )
    except Exception as exc:
        _log("Skip dynamic warmup prompts: tokenizer load failed: %r", exc)
        return []

    use_chat_template = _env_enabled("MS_INFERRT_WARMUP_USE_CHAT_TEMPLATE", "1")
    lengths: list[int] = []
    for text in texts:
        try:
            if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
                token_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                lengths.append(len(token_ids))
            else:
                encoded = tokenizer(text, add_special_tokens=True)
                lengths.append(len(encoded["input_ids"]))
        except Exception as exc:
            _log("Ignore prompt during dynamic warmup tokenization: %r", exc)
    return [length for length in lengths if length > 0]


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _range_record(assigned: list[tuple[int, int]], bucket: int,
                  source: str) -> dict[str, int | str]:
    effective_values = [effective for effective, _ in assigned]
    prompt_values = [original for _, original in assigned]
    return {
        "range": f"{min(effective_values)}-{max(effective_values)}",
        "min": min(effective_values),
        "max": max(effective_values),
        "bucket": bucket,
        "count": len(assigned),
        "prompt_min": min(prompt_values),
        "prompt_max": max(prompt_values),
        "source": source,
    }


def _rounded_unique_buckets(effective_pairs: list[tuple[int, int]],
                            granularity: int, max_buckets: int
                            ) -> tuple[list[int], list[dict[str, int | str]]] | None:
    buckets = sorted({_ceil_to_multiple(item[0], granularity)
                      for item in effective_pairs})
    if len(buckets) > max_buckets:
        return None
    ranges: list[dict[str, int | str]] = []
    lower = 1
    for bucket in buckets:
        assigned = [item for item in effective_pairs if lower <= item[0] <= bucket]
        if assigned:
            ranges.append(_range_record(
                assigned, bucket, "prompt_lengths_rounded_unique"))
        lower = bucket + 1
    return _unique_sorted_desc(buckets), ranges


def _quantile_buckets(effective_pairs: list[tuple[int, int]], granularity: int,
                      max_buckets: int, max_tokens: int | None
                      ) -> tuple[list[int], list[dict[str, int | str]]]:
    filtered = [effective for effective, _ in effective_pairs]
    buckets: list[int] = []
    for idx in range(1, max_buckets + 1):
        pos = min(math.ceil(len(filtered) * idx / max_buckets) - 1,
                  len(filtered) - 1)
        bucket = _ceil_to_multiple(filtered[max(pos, 0)], granularity)
        buckets.append(min(bucket, max_tokens) if max_tokens is not None else bucket)
    buckets = sorted(set(buckets))
    ranges: list[dict[str, int | str]] = []
    start = 0
    for bucket in buckets:
        assigned = [item for item in effective_pairs[start:] if item[0] <= bucket]
        if assigned:
            ranges.append(_range_record(assigned, bucket, "prompt_lengths_quantile"))
            start += len(assigned)
    return _unique_sorted_desc(buckets), ranges


def _dynamic_buckets_from_lengths(
    lengths: list[int],
    max_tokens: int | None,
) -> tuple[list[int], list[dict[str, int | str]]]:
    if not lengths:
        return [], []

    max_buckets = _env_int("MS_INFERRT_WARMUP_MAX_BUCKETS", 6)
    granularity = _env_int("MS_INFERRT_WARMUP_BUCKET_GRANULARITY", 128)
    strategy = os.environ.get("MS_INFERRT_WARMUP_DYNAMIC_STRATEGY", "quantile").strip().lower()

    # vLLM chunked prefill may split a long prompt into runtime chunks bounded
    # by max_num_batched_tokens. Keep original prompt lengths in range metadata,
    # but warm the executable chunk shape for prompts above that bound.
    effective_pairs = sorted(
        (
            min(length, max_tokens) if max_tokens is not None else length,
            length,
        )
        for length in lengths
        if length > 0
    )
    if not effective_pairs:
        return [], []
    if strategy == "rounded_unique":
        rounded = _rounded_unique_buckets(
            effective_pairs, granularity, max_buckets)
        if rounded is not None:
            return rounded
    return _quantile_buckets(
        effective_pairs, granularity, max_buckets, max_tokens)


def _report_dir() -> Path:
    return Path(
        os.environ.get(
            "MS_INFERRT_WARMUP_REPORT_DIR",
            os.environ.get(
                "MS_INFERRT_INTERNAL_REPORT_DIR",
                str(RUNTIME_DIR / "inferrt_warmup_reports"),
            ),
        )
    )


def _write_warmup_plan(
    buckets: list[int],
    ranges: list[dict[str, int | str]],
    max_tokens: int | None,
    prompt_lengths: list[int] | None = None,
    failures: list[dict[str, str]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if not _env_enabled("MS_INFERRT_WARMUP_REPORT", "1"):
        return
    out_dir = _report_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "buckets_desc": buckets,
        "ranges": ranges,
        "source": os.environ.get("MS_INFERRT_WARMUP_PREFILL_SOURCE", "manual"),
        "range_policy": os.environ.get("MS_INFERRT_WARMUP_RANGE_POLICY", "upper"),
        "max_dummy_tokens": max_tokens,
        "prompt_length_count": len(prompt_lengths or []),
        "prompt_length_min": min(prompt_lengths) if prompt_lengths else None,
        "prompt_length_max": max(prompt_lengths) if prompt_lengths else None,
        "failures": failures or [],
        "metadata": metadata or {},
        "purpose": (
            "Precompile representative prefill graphs for configured input "
            "length buckets before the service handles real requests."
        ),
    }
    path = out_dir / f"warmup_bucket_plan_pid{os.getpid()}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_candidate_lengths(value: str) -> list[int]:
    return _unique_sorted_desc(_parse_int_list(value))


def _profile_adaptive_candidates(model_runner: Any) -> list[int]:
    raw = os.environ.get("MS_INFERRT_WARMUP_PROFILE_CANDIDATES", "").strip()
    if raw:
        return sorted(_parse_candidate_lengths(raw))

    max_tokens = _max_dummy_tokens(model_runner)
    model_max_len = _model_max_len(model_runner)
    upper = max_tokens or model_max_len or 4096
    if model_max_len is not None and max_tokens is not None:
        upper = min(model_max_len, max_tokens)

    try:
        min_bucket = max(1, int(os.environ.get("MS_INFERRT_WARMUP_PROFILE_MIN_BUCKET", "128")))
    except ValueError:
        min_bucket = 128
    try:
        max_candidates = max(2, int(os.environ.get("MS_INFERRT_WARMUP_PROFILE_MAX_CANDIDATES", "12")))
    except ValueError:
        max_candidates = 12

    candidates: set[int] = set()
    value = min_bucket
    while value <= upper:
        candidates.add(value)
        if value < 1024:
            value += 256
        elif value < 4096:
            value += 512
        else:
            value += 1024
    candidates.add(upper)
    candidates = {max(1, min(int(item), int(upper))) for item in candidates}
    ordered = sorted(candidates)
    if len(ordered) <= max_candidates:
        return ordered

    # Keep endpoints and sample the middle evenly. Candidate probing is real
    # compilation work, so cap it separately from the final selected buckets.
    selected = {ordered[0], ordered[-1]}
    for idx in range(1, max_candidates - 1):
        pos = round((len(ordered) - 1) * idx / (max_candidates - 1))
        selected.add(ordered[pos])
    return sorted(selected)


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _score_profile_probes(successful: list[dict[str, Any]],
                          max_tokens: int | None) -> dict[int, dict[str, Any]]:
    jump_ratio = _env_float("MS_INFERRT_WARMUP_PROFILE_JUMP_RATIO", 0.18)
    slope_ratio = _env_float("MS_INFERRT_WARMUP_PROFILE_SLOPE_RATIO", 0.45)
    selected: dict[int, dict[str, Any]] = {}

    def add(probe: dict[str, Any], reason: str, score: float = 1.0) -> None:
        bucket = int(probe["bucket"])
        item = selected.setdefault(
            bucket,
            {
                "bucket": bucket,
                "reasons": [],
                "score": 0.0,
                "elapsed_ms": float(probe["elapsed_ms"]),
            },
        )
        if reason not in item["reasons"]:
            item["reasons"].append(reason)
        item["score"] += score

    add(successful[0], "lower_boundary", 1.0)
    add(successful[-1], "upper_or_chunk_boundary", 2.0)
    if max_tokens is not None:
        for probe in successful:
            if int(probe["bucket"]) == max_tokens:
                add(probe, "max_num_batched_tokens_boundary", 2.0)

    prev = successful[0]
    prev_slope: float | None = None
    for probe in successful[1:]:
        bucket = int(probe["bucket"])
        prev_bucket = int(prev["bucket"])
        elapsed = float(probe["elapsed_ms"])
        prev_elapsed = float(prev["elapsed_ms"])
        if elapsed >= prev_elapsed * (1.0 + jump_ratio):
            add(probe, "latency_jump", min(3.0, elapsed / max(prev_elapsed, 1e-6)))
        delta_tokens = max(1, bucket - prev_bucket)
        slope = (elapsed - prev_elapsed) / delta_tokens
        if prev_slope is not None and abs(prev_slope) > 1e-9:
            if abs(slope - prev_slope) / abs(prev_slope) >= slope_ratio:
                add(probe, "slope_change", 1.5)
        prev_slope = slope
        prev = probe
    return selected


def _profile_ranges(successful: list[dict[str, Any]], buckets: list[int],
                    selected: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    elapsed_by_bucket = {int(item["bucket"]): float(item["elapsed_ms"])
                         for item in successful}
    ranges: list[dict[str, Any]] = []
    lower = 1
    for bucket in buckets:
        covered = [int(item["bucket"]) for item in successful
                   if lower <= int(item["bucket"]) <= bucket]
        ranges.append({
            "range": f"{lower}-{bucket}",
            "min": lower,
            "max": bucket,
            "bucket": bucket,
            "count": len(covered),
            "source": "profile_adaptive",
            "reasons": selected.get(bucket, {}).get("reasons", ["coverage"]),
            "probe_elapsed_ms": round(elapsed_by_bucket.get(bucket, 0.0), 3),
        })
        lower = bucket + 1
    return ranges


def _selected_profile_adaptive_ranges(
    probes: list[dict[str, Any]],
    max_tokens: int | None,
) -> tuple[list[int], list[dict[str, Any]]]:
    successful = sorted(
        (probe for probe in probes
         if probe.get("ok") and isinstance(probe.get("elapsed_ms"), (int, float))),
        key=lambda item: int(item["bucket"]),
    )
    if not successful:
        return [], []

    max_buckets = _env_int("MS_INFERRT_WARMUP_MAX_BUCKETS", 6)
    selected = _score_profile_probes(successful, max_tokens)

    ranked = sorted(
        selected.values(),
        key=lambda item: (float(item["score"]), float(item["elapsed_ms"]), int(item["bucket"])),
        reverse=True,
    )
    kept = sorted(ranked[:max_buckets], key=lambda item: int(item["bucket"]))
    buckets_asc = [int(item["bucket"]) for item in kept]
    if successful[-1]["bucket"] not in buckets_asc:
        buckets_asc.append(int(successful[-1]["bucket"]))
    buckets_asc = sorted(set(buckets_asc))
    return _unique_sorted_desc(buckets_asc), _profile_ranges(
        successful, buckets_asc, selected)


def _probe_profile_adaptive_buckets(
    model_runner: Any,
    cudagraph_mode: Any,
    force_attention: bool,
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    candidates = _profile_adaptive_candidates(model_runner)
    max_tokens = _max_dummy_tokens(model_runner)
    if max_tokens is not None:
        candidates = [candidate for candidate in candidates if candidate <= max_tokens]

    probes: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    _log("InferRT profile-adaptive warmup probe candidates: %s.", candidates)
    for bucket in candidates:
        started = time.perf_counter()
        previous_phase = os.environ.get("MS_INFERRT_WARMUP_ACTIVE_PHASE")
        previous_bucket = os.environ.get("MS_INFERRT_WARMUP_ACTIVE_BUCKET")
        os.environ["MS_INFERRT_WARMUP_ACTIVE_PHASE"] = "profile_probe"
        os.environ["MS_INFERRT_WARMUP_ACTIVE_BUCKET"] = str(bucket)
        try:
            model_runner._dummy_run(
                bucket,
                with_prefill=True,
                cudagraph_runtime_mode=cudagraph_mode,
                force_attention=force_attention,
                is_profile=False,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            probes.append(
                {
                    "bucket": bucket,
                    "ok": True,
                    "elapsed_ms": round(elapsed_ms, 3),
                }
            )
            _log(
                "InferRT profile-adaptive probe bucket=%d elapsed_ms=%.3f.",
                bucket,
                elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            error = repr(exc)
            probes.append(
                {
                    "bucket": bucket,
                    "ok": False,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "error": error,
                }
            )
            failures.append({"bucket": str(bucket), "error": error})
            if not _env_enabled("MS_INFERRT_WARMUP_IGNORE_ERRORS", "1"):
                raise
            _log(
                "InferRT profile-adaptive probe bucket=%d failed and was ignored: %r",
                bucket,
                exc,
            )
        finally:
            if previous_phase is None:
                os.environ.pop("MS_INFERRT_WARMUP_ACTIVE_PHASE", None)
            else:
                os.environ["MS_INFERRT_WARMUP_ACTIVE_PHASE"] = previous_phase
            if previous_bucket is None:
                os.environ.pop("MS_INFERRT_WARMUP_ACTIVE_BUCKET", None)
            else:
                os.environ["MS_INFERRT_WARMUP_ACTIVE_BUCKET"] = previous_bucket

    buckets, ranges = _selected_profile_adaptive_ranges(probes, max_tokens)
    metadata = {
        "profile_adaptive": True,
        "model_max_len": _model_max_len(model_runner),
        "max_dummy_tokens": max_tokens,
        "probe_candidates": candidates,
        "probes": probes,
        "selection_policy": {
            "max_buckets": os.environ.get("MS_INFERRT_WARMUP_MAX_BUCKETS", "6"),
            "jump_ratio": os.environ.get("MS_INFERRT_WARMUP_PROFILE_JUMP_RATIO", "0.18"),
            "slope_ratio": os.environ.get("MS_INFERRT_WARMUP_PROFILE_SLOPE_RATIO", "0.45"),
        },
    }
    return buckets, ranges, failures, metadata


def _filtered_prefill_buckets(model_runner: Any) -> tuple[list[int], list[dict[str, int | str]], int | None]:
    _filtered_prefill_buckets.prompt_lengths = []  # type: ignore[attr-defined]
    ranges: list[dict[str, int | str]] = []
    raw_buckets = os.environ.get(
        "MS_INFERRT_WARMUP_PREFILL_BUCKETS",
        os.environ.get("MS_INFERRT_WARMUP_PREFILL_SHAPES", ""),
    )
    raw_ranges = os.environ.get("MS_INFERRT_WARMUP_PREFILL_RANGES", "")
    source = os.environ.get("MS_INFERRT_WARMUP_PREFILL_SOURCE", "").strip().lower()
    max_tokens = _max_dummy_tokens(model_runner)
    if source in ("prompts", "prompt", "dataset", "auto"):
        lengths = _token_lengths_for_prompts(model_runner)
        buckets, ranges = _dynamic_buckets_from_lengths(lengths, max_tokens)
        if buckets:
            _filtered_prefill_buckets.prompt_lengths = lengths  # type: ignore[attr-defined]
            return buckets, ranges, max_tokens
        _log(
            "Dynamic prompt-length warmup produced no valid buckets; falling "
            "back to manual ranges/buckets if provided."
        )

    if raw_ranges.strip():
        buckets, ranges = _parse_prefill_ranges(raw_ranges)
    elif raw_buckets.strip():
        buckets = _parse_int_list(raw_buckets)
        ranges = [
            {"range": str(bucket), "min": bucket, "max": bucket, "bucket": bucket}
            for bucket in buckets
        ]
    else:
        _log(
            "Skip InferRT extra warmup: neither "
            "MS_INFERRT_WARMUP_PREFILL_RANGES nor "
            "MS_INFERRT_WARMUP_PREFILL_BUCKETS is set. Extra prefill warmup "
            "is intentionally explicit because fixed default buckets can "
            "regress long chunked-prefill workloads."
        )
        return [], [], max_tokens

    if max_tokens is not None:
        dropped = [bucket for bucket in buckets if bucket > max_tokens]
        if dropped:
            _log(
                "Drop InferRT extra warmup buckets above max dummy tokens %s: %s",
                max_tokens,
                dropped,
            )
        buckets = [bucket for bucket in buckets if bucket <= max_tokens]
        ranges = [
            item
            for item in ranges
            if int(item.get("bucket", 0)) <= max_tokens
        ]
    buckets = _unique_sorted_desc(buckets)
    return buckets, ranges, max_tokens


def _restore_active_warmup(previous_phase: str | None, previous_bucket: str | None) -> None:
    for name, value in (
        ("MS_INFERRT_WARMUP_ACTIVE_PHASE", previous_phase),
        ("MS_INFERRT_WARMUP_ACTIVE_BUCKET", previous_bucket),
    ):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _warm_bucket(model_runner: Any, bucket: int, cudagraph_mode: Any,
                 force_attention: bool, failures: list[dict[str, str]]) -> None:
    previous_phase = os.environ.get("MS_INFERRT_WARMUP_ACTIVE_PHASE")
    previous_bucket = os.environ.get("MS_INFERRT_WARMUP_ACTIVE_BUCKET")
    os.environ["MS_INFERRT_WARMUP_ACTIVE_PHASE"] = "selected_warmup"
    os.environ["MS_INFERRT_WARMUP_ACTIVE_BUCKET"] = str(bucket)
    try:
        _log(
            "InferRT bucketed prefill warmup dummy_run: "
            "compile_bucket_tokens=%d, with_prefill=True, force_attention=%s.",
            bucket,
            force_attention,
        )
        model_runner._dummy_run(
            bucket,
            with_prefill=True,
            cudagraph_runtime_mode=cudagraph_mode,
            force_attention=force_attention,
            is_profile=False,
        )
    except Exception as exc:
        failures.append({"bucket": str(bucket), "error": repr(exc)})
        if not _env_enabled("MS_INFERRT_WARMUP_IGNORE_ERRORS", "1"):
            raise
        _log("InferRT extra prefill warmup bucket %d failed and was ignored: %r",
             bucket, exc)
    finally:
        _restore_active_warmup(previous_phase, previous_bucket)


def _run_extra_warmup(model_runner: Any, cudagraph_mode: Any) -> None:
    force_attention = _env_enabled("MS_INFERRT_WARMUP_FORCE_ATTENTION", "0")
    source = os.environ.get("MS_INFERRT_WARMUP_PREFILL_SOURCE", "").strip().lower()
    metadata: dict[str, Any] = {}
    prompt_lengths: list[int] = []
    failures: list[dict[str, str]] = []
    max_tokens = _max_dummy_tokens(model_runner)
    if source in ("profile_adaptive", "adaptive_profile", "profile"):
        buckets, ranges, probe_failures, metadata = _probe_profile_adaptive_buckets(
            model_runner, cudagraph_mode, force_attention)
        failures.extend(probe_failures)
    else:
        buckets, ranges, max_tokens = _filtered_prefill_buckets(model_runner)
        prompt_lengths = getattr(_filtered_prefill_buckets, "prompt_lengths", [])

    _write_warmup_plan(buckets, ranges, max_tokens, prompt_lengths, failures, metadata)
    if not buckets:
        _log("Skip InferRT extra warmup: no valid prefill buckets.")
        return
    _log("InferRT bucketed prefill warmup plan: ranges=%s, buckets=%s.",
         ranges, buckets)
    for bucket in buckets:
        _warm_bucket(model_runner, bucket, cudagraph_mode, force_attention, failures)

    os.environ["MS_INFERRT_WARMUP_COMPLETED"] = "1"
    _write_warmup_plan(buckets, ranges, max_tokens, prompt_lengths, failures, metadata)
    if failures:
        _log("InferRT extra warmup completed with ignored failures: %s", failures)
    else:
        _log("InferRT bucketed prefill warmup completed for ranges: %s", ranges)


def patch_vllm_ascend_warmup() -> None:
    if not _env_enabled("MS_INFERRT_PATCH_WARMUP", "0"):
        return

    try:
        from vllm.config import CUDAGraphMode
        from vllm_ascend.worker.worker import NPUWorker
    except Exception as exc:
        _log("Skip InferRT extra warmup patch: %s", exc)
        return

    current = NPUWorker.compile_or_warm_up_model
    if getattr(current, "_inferrt_extra_warmup_patched", False):
        return

    original_compile_or_warm_up_model = current

    def compile_or_warm_up_model(self: Any) -> None:
        original_compile_or_warm_up_model(self)
        if not _env_enabled("MS_INFERRT_PATCH_WARMUP", "0"):
            return
        model_runner = getattr(self, "model_runner", None)
        if model_runner is None:
            _log("Skip InferRT extra warmup: worker has no model_runner.")
            return
        _run_extra_warmup(model_runner, CUDAGraphMode.NONE)

    compile_or_warm_up_model._inferrt_extra_warmup_patched = True  # type: ignore[attr-defined]
    NPUWorker.compile_or_warm_up_model = compile_or_warm_up_model
    _log("Patched vLLM-Ascend NPUWorker.compile_or_warm_up_model for InferRT.")


__all__ = ["patch_vllm_ascend_warmup"]
