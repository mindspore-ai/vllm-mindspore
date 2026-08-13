# InferRT vLLM Backend Patch

This directory provides an external integration that uses InferRT as vLLM's
prefill `torch.compile` backend without modifying vLLM, vLLM-Ascend, or
installed `site-packages`.

## Execution Policy

- Prefill uses `ms_inferrt.torch.backend` at the same backend-selection layer
  where vLLM would otherwise select inductor.
- Prefill uses direct Dynamo tracing with `splitting_ops=[]` and does not use
  piecewise or ACLGraph execution.
- Decode is explicitly marked and dispatched to vLLM's original backend. On
  vLLM-Ascend it may use `FULL_DECODE_ONLY` ACLGraph capture and replay.
- Shared model graphs first seen during decode capture still retain an InferRT
  callable for later prefill execution; runtime decode calls use the original
  callable.

The resolved engine configuration should contain:

```text
backend='inferrt'
splitting_ops=[]
use_inductor_graph_partition=False
cudagraph_mode=<CUDAGraphMode.FULL_DECODE_ONLY: (2, 0)>
```

## Files

- `inferrt_vllm_backend_launcher.py`: user-facing vLLM launcher.
- `sitecustomize.py`: applies the patch in worker subprocesses.
- `inferrt_patch.py`: backend replacement, compilation configuration, and
  explicit prefill/decode stage routing.
- `backend_optimizer.py`: wraps `ms_inferrt.torch.backend`, applies guarded FX
  rewrites, and records graph compilation decisions.
- `qwen35_compat.py`: required Qwen3.5 capture compatibility patches.

The optional AI-search warmup is provided by the sibling `../warmup` directory.
When present, the launcher adds its patch directory to worker `PYTHONPATH` and
the backend patch installs its worker-level warmup hook.

## Launch

```bash
export ASCEND_ENV=/path/to/Ascend/cann-9.0.0/set_env.sh
export PYTHON=/path/to/conda/env/bin/python
export MODEL_PATH=/path/to/model
export REGISTER_ROOT=/path/to/repo/inferrt/python/ms_inferrt/external_patch/register

source "${ASCEND_ENV}"
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_PLUGINS=ascend
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

Expected log markers include:

```text
Using optimized ms_inferrt.torch.backend as vLLM torch.compile backend
Configured direct InferRT torch.compile backend
Capturing CUDA graphs (decode, FULL)
Replaying aclgraph
```

## Python Module Details

### `inferrt_vllm_backend_launcher.py`

**Purpose:** This is the user-facing entry point for launching vLLM with the
external InferRT integration. It replaces source-tree and `site-packages`
edits with process-local monkey patching.

**Responsibilities:**

1. Sets the environment switches required by the direct InferRT backend.
2. Adds `register` and the optional sibling `warmup` directory to `sys.path`.
3. Propagates both directories through `PYTHONPATH` so worker subprocesses load
   the same integration through `sitecustomize.py`.
4. Calls `inferrt_patch.apply_patch()` before importing the vLLM CLI.
5. Transfers control to `vllm.entrypoints.cli.main`.

`PATCH_DIR` resolves to the current `register` directory, while
`WARMUP_PATCH_DIR` resolves to `../warmup`. The launcher belongs to both the
backend-registration path and the managed-warmup launch path.

### `inferrt_patch.py`

**Purpose:** This module installs the external vLLM monkey patches that route
prefill through InferRT and preserve native vLLM decode behavior.

**Responsibilities:**

1. Patches `CompilationConfig.init_backend` to return the wrapped
   `ms_inferrt.torch.backend`.
2. Forces the direct prefill configuration: `backend='inferrt'`,
   `splitting_ops=[]`, and `use_inductor_graph_partition=False`.
3. Marks prefill and decode execution explicitly instead of inferring the
   stage from graph size or operator counts.
4. Routes decode to the original vLLM backend and permits
   `FULL_DECODE_ONLY` ACLGraph capture and replay.
5. Protects tensor-parallel communication and reduction paths from unsafe
   identity rewrites.
6. Applies required Qwen3.5 compatibility patches.
7. Loads `warmup_patch.patch_vllm_ascend_warmup()` when the sibling warmup
   module is available and enabled.

Important entry points include `patch_compilation_backend()`,
`patch_vllm_ascend_config()`, `patch_vllm_mindspore_config()`, stage-marker
patches for model execution and ACLGraph capture/replay, and `apply_patch()`.
This is the central backend-registration module.

### `backend_optimizer.py`

**Purpose:** This is a thin observable wrapper around the real
`ms_inferrt.torch.backend`; it does not copy or replace InferRT FX lowering.

**Responsibilities:**

1. Collects FX graph statistics and stable graph signatures.
2. Records compile and cache events in `graph_compile_cache_pid*.jsonl`.
3. Applies narrowly guarded FX compatibility rewrites.
4. Recursively validates ordinary and immutable FX input containers without
   treating graph metadata as unsupported runtime input.
5. Builds both an InferRT callable and, when required, the original-backend
   callable for stage-aware runtime dispatch.
6. Sends prefill calls to InferRT and decode calls to the original vLLM
   backend.

Key functions include `_graph_stats()`, `_graph_signature()`,
`_record_compile_cache_event()`, `optimize_graph()`,
`_should_use_inferrt_for_graph()`, `_runtime_dispatch_prefill_only()`, and
`make_backend()`. The compilation reports produced here are also the feedback
source used by AI warmup verification.

### `qwen35_compat.py`

**Purpose:** This module contains required Qwen3.5 compatibility fixes for
graph capture. These are correctness and availability fixes, not optional
performance experiments.

**Responsibilities:**

1. Keeps unsupported GDN-specific kernels outside the InferRT-captured FX
   region where necessary.
2. Selects the native gated RMSNorm path when the out-of-tree path produces
   unsupported capture metadata.
3. Prevents Triton, higher-order operator, or non-serializable metadata from
   becoming invalid InferRT graph inputs.

`patch_qwen35_gdn_capture_guard()` and
`patch_qwen35_gated_rmsnorm_native()` implement the targeted fixes;
`apply_qwen35_compat_patches()` is the integration entry point. This module is
required for Qwen3.5 but is normally inert for unrelated model families.

### `sitecustomize.py`

**Purpose:** Python imports a `sitecustomize.py` found on `PYTHONPATH` during
interpreter startup. This module ensures that spawned vLLM workers receive the
same patch as the launcher process.

**Responsibilities:**

1. Bridges compatible Triton API differences, including a missing
   `constexpr_function` entry point.
2. Avoids duplicate torch-npu synchronization trace-rule registration.
3. Calls `inferrt_patch.apply_patch()` when the external integration is
   enabled.
4. Tolerates partially installed optional dependencies during early Python
   startup.

Without this module, a multiprocess or tensor-parallel launch could patch only
the parent process while leaving workers on the default backend.
