# AI Warmup Integration Changes

The warmup layer builds on the direct InferRT backend in `../register`. It adds
AI-guided warmup without changing InferRT operator implementations or enabling
piecewise compilation, prefill ACLGraph execution, experimental patches, or
the legacy `CompilerInterface` path.

## Backend Integration Points

- `../register/inferrt_vllm_backend_launcher.py`
  - Adds the sibling `warmup` directory to `sys.path` and worker
    `PYTHONPATH` when it is present.
- `../register/inferrt_patch.py`
  - Calls `patch_vllm_ascend_warmup()` when the optional warmup module is
    available and enabled.
- `../register/backend_optimizer.py`
  - Writes graph compile events consumed by warmup verification.

The detailed documentation for these modules is consolidated in
`../register/README.md`.

## Warmup Files

- `warmup_patch.py`
- `service_warmup_manager.py`
- `managed_warmup_launcher.py`
- `README.md`
- `AI_WARMUP_DIRECT_INFERRT_USAGE.md`
- `CODE_STRUCTURE.md`
- `.gitignore`

Detailed Python module documentation is consolidated in this directory's
`README.md`.

## Explicitly Excluded

- `experimental_patches.py`
- `inferrt_compiler.py`
- custom-operator stubs
- benchmark matrix tools
- full Conda environment exports

## Preserved Execution Policy

- Prefill replaces the inductor-layer backend with InferRT and does not use
  piecewise or ACLGraph execution.
- Decode uses the native vLLM backend and may use `FULL_DECODE_ONLY` ACLGraph.
- Warmup completes before user traffic is marked ready, and synthetic warmup
  requests remain separate from measured real requests.
