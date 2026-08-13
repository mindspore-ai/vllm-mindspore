# InferRT Direct Backend Warmup Integration

This folder contains AI warmup orchestration. The InferRT backend replacement
lives in the sibling `../register` directory.

## Goal

- Use InferRT as the direct `torch.compile` backend selected at the inductor
  backend layer for prefill.
- Do not use vLLM-Ascend ACLGraph or piecewise execution for prefill.
- Keep decode on the original vLLM backend path. Decode may use the native
  FULL_DECODE_ONLY ACLGraph path and should not enter InferRT unless explicitly
  requested.
- Keep vLLM and installed site-packages unmodified. The integration is applied
  by a launcher-side monkey patch.
- Run optional HTTP-level AI warmup before user traffic, so real requests reuse
  already compiled InferRT graphs whenever possible.

## Main Files

- `../register/inferrt_vllm_backend_launcher.py`
  - User-facing launcher.
  - Adds the backend patch and optional warmup patch directories to `PYTHONPATH`.
  - Applies the external patch before vLLM creates the engine.

- `../register/inferrt_patch.py`
  - Forces direct InferRT backend config for prefill.
  - Clears prefill piecewise split and prevents prefill ACLGraph fallback.
  - Marks decode execution/capture as decode so it can use the original vLLM
    backend and native ACLGraph path.

- `../register/backend_optimizer.py`
  - Wraps `ms_inferrt.torch.backend`.
  - Applies guarded graph rewrites and reporting used by the current optimized
    direct backend path.
  - Dispatches by explicit runtime stage: prefill uses InferRT, decode uses the
    original backend unless `MS_INFERRT_COMPILE_DECODE=1`.

- `service_warmup_manager.py`
  - Sends AI-selected fake HTTP requests to cover likely graph/shape paths.
  - Records graph compile reports and checks whether real requests recompile.

- `managed_warmup_launcher.py`
  - Starts vLLM with the direct InferRT launcher.
  - Waits for health, runs warmup, then writes a ready file.
  - Use this when the user should not see the warmup stage.

## Required Environment Variables

Set these paths for your own machine before running the examples:

```bash
export ASCEND_ENV=/path/to/Ascend/cann-9.0.0/set_env.sh
export PYTHON=/path/to/conda/env/bin/python
export MODEL_PATH=/path/to/Qwen3-8B
export ASCEND_RT_VISIBLE_DEVICES=0
export EXTERNAL_PATCH_ROOT=/path/to/repo/inferrt/python/ms_inferrt/external_patch
export WARMUP_ROOT=${EXTERNAL_PATCH_ROOT}/warmup
export REGISTER_ROOT=${EXTERNAL_PATCH_ROOT}/register
```

## Basic Direct Backend Launch

```bash
source "${ASCEND_ENV}"
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_PLUGINS=ascend
export VLLM_TORCH_COMPILE_BACKEND=inferrt
export INFERRT_VLLM_EXTERNAL_PATCH=1
export MS_INFERRT_DECODE_USE_ORIGINAL_BACKEND=1
export MS_INFERRT_COMPILE_DECODE=0

"${PYTHON}" "${REGISTER_ROOT}/inferrt_vllm_backend_launcher.py" \
  serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 22000 \
  --max-num-batched-tokens 4096 \
  --compilation-config '{"backend":"inferrt"}'
```

Expected log signs for prefill:

- `Using optimized ms_inferrt.torch.backend as vLLM torch.compile backend`
- `Configured direct InferRT torch.compile backend`
- `splitting_ops=[]`
- `use_inductor_graph_partition=False`

Expected log signs for decode:

- `Capturing CUDA graphs (decode, FULL)`
- `Replaying aclgraph`

The prefill path should not log:

- `PIECEWISE compilation enabled`
- `using only ACL Graph mode`
- `Calculated maximum supported batch sizes for ACL graph`

## Managed Warmup Launch

```bash
source "${ASCEND_ENV}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_PLUGINS=ascend
export VLLM_TORCH_COMPILE_BACKEND=inferrt
export INFERRT_VLLM_EXTERNAL_PATCH=1

"${PYTHON}" "${WARMUP_ROOT}/managed_warmup_launcher.py" \
  --ready-file "${WARMUP_ROOT}/runtime"/qwen3_inferrt.ready \
  --report-dir "${WARMUP_ROOT}/runtime"/qwen3_inferrt_warmup_reports \
  --server-log "${WARMUP_ROOT}/runtime"/qwen3_inferrt_server.log \
  --warmup-lengths auto \
  --coverage-policy broad \
  --warmup-rounds 2 \
  --warmup-max-tokens 1 \
  --warmup-decode-max-tokens 1 \
  --warmup-batch-sizes 1,2,4,8 \
  --warmup-verify-policy adaptive \
  --worker-warmup-source profile_adaptive \
  --worker-warmup-profile-max-candidates 12 \
  --cache-miss-policy report \
  -- \
  serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 22000 \
  --max-num-batched-tokens 4096 \
  --tensor-parallel-size 4 \
  --compilation-config '{"backend":"inferrt"}'
```

Traffic should wait for the ready file. This keeps warmup hidden from normal
users while preserving direct InferRT backend behavior.

## Python Module Details

### `managed_warmup_launcher.py`

**Purpose:** This is the managed service entry point. It starts the direct
InferRT vLLM launcher, waits for service health, runs warmup, and exposes
readiness only after warmup completes.

**Responsibilities:**

1. Parses vLLM server arguments such as model, host, port, model length, and
   maximum batched tokens.
2. Locates the sibling backend launcher at
   `../register/inferrt_vllm_backend_launcher.py`.
3. Derives automatic warmup lengths from model and chunk limits.
4. Starts the vLLM server in a managed process group and captures its log.
5. Invokes `service_warmup_manager.py` after the health endpoint responds.
6. Writes a ready file only after the configured warmup and verification
   policy succeeds.
7. Forwards signals and terminates the managed process group cleanly.

Important helpers include `_parse_server_value()`, `_parse_model()`,
`_auto_warmup_lengths()`, `_server_environment()`, `_start_server()`, and
`main()`. This module is the recommended entry point when users must not
observe an un-warmed service.

### `service_warmup_manager.py`

**Purpose:** This module performs service-level warmup through the
OpenAI-compatible HTTP API, exercising the same tokenizer, scheduler, chunked
prefill, and backend path used by real requests.

**Responsibilities:**

1. Builds candidate lengths according to the requested coverage policy.
2. Uses the model tokenizer to construct prompts with controlled token counts.
3. Builds warmup and verification request plans with separate request data.
4. Sends request groups concurrently and records request outcomes.
5. Reads graph compilation reports written by `backend_optimizer.py`.
6. Uses compile signatures and cache events as feedback to identify uncovered
   graph regions.
7. Writes a machine-readable warmup report containing coverage and timing
   results.

Key functions include `coverage_lengths()`, `representative_prompts()`,
`build_request_plan()`, `build_verify_plan()`, `wait_health()`,
`build_prompt()`, `send_chat_request()`, `send_request_group()`,
`read_compile_reports()`, and `summarize_compile_rows()`.

### `warmup_patch.py`

**Purpose:** This module adds optional worker-level prefill probes after
vLLM-Ascend's native `NPUWorker.compile_or_warm_up_model()` phase.

**Responsibilities:**

1. Reads manual buckets, ranges, prompt distributions, or profile-adaptive
   candidate settings.
2. Uses `model_runner._dummy_run()` to trigger prefill graph compilation before
   user traffic.
3. Adapts candidate selection to model length and
   `max_num_batched_tokens` limits.
4. Measures probe latency and selects regions around timing or slope changes.
5. Writes `warmup_bucket_plan_pid*.json` with the selected warmup plan.

Important helpers include `_max_dummy_tokens()`, `_model_max_len()`,
`_token_lengths_for_prompts()`, `_dynamic_buckets_from_lengths()`,
`_profile_adaptive_candidates()`, `_probe_profile_adaptive_buckets()`,
`_selected_profile_adaptive_ranges()`, and `_filtered_prefill_buckets()`.
`patch_vllm_ascend_warmup()` installs the worker hook. The feature is opt-in;
the direct InferRT backend remains usable when worker warmup is disabled.
