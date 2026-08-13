# InferRT AI Warmup Code Structure

This document defines the file boundary of the AI warmup integration. The
warmup layer depends on the direct InferRT backend in the sibling `register`
directory. It does not include benchmark tools, experimental operator patches,
or the legacy `CompilerInterface` adapter.

## Functional Boundary

- Prefill is compiled and executed by `ms_inferrt.torch.backend` through the
  direct backend registration layer.
- Decode is dispatched to the original vLLM backend and may use
  `FULL_DECODE_ONLY` ACLGraph capture and replay.
- Warmup sends synthetic requests before the service becomes ready and uses
  compilation feedback to improve graph coverage.
- Real-request reporting detects graph compilations that occur after warmup.

## Backend Registration Files

| File | Responsibility |
|---|---|
| `../register/inferrt_vllm_backend_launcher.py` | Loads the backend and optional warmup patches before vLLM starts. |
| `../register/inferrt_patch.py` | Installs backend routing and the optional worker warmup hook. |
| `../register/backend_optimizer.py` | Records graph compilation events during warmup and real traffic. |

## Warmup Files

| File | Responsibility |
|---|---|
| `warmup_patch.py` | Adds configurable synthetic prefill runs after native worker warmup. |
| `service_warmup_manager.py` | Selects requests, sends HTTP warmup traffic, reads compile feedback, and verifies coverage. |
| `managed_warmup_launcher.py` | Starts the service, waits for health, runs warmup, and writes the ready file. |
| `README.md` | Quick start plus detailed Python module documentation. |
| `AI_WARMUP_DIRECT_INFERRT_USAGE.md` | Complete configuration, validation, and troubleshooting guide. |
| `INTEGRATION_CHANGES.md` | Warmup changes relative to the backend registration layer. |

## Call Chain

```text
managed_warmup_launcher.py
  -> ../register/inferrt_vllm_backend_launcher.py
  -> inferrt_patch.apply_patch()
  -> warmup_patch.patch_vllm_ascend_warmup()
  -> vLLM service becomes healthy
  -> service_warmup_manager.py sends selected synthetic requests
  -> graph compile reports provide feedback
  -> ready file is written
  -> real traffic starts
```

## Acceptance Criteria

1. Logs report the `inferrt` backend and `splitting_ops=[]`.
2. Prefill emits InferRT IR and does not enter piecewise or ACLGraph execution.
3. Decode logs include `Capturing CUDA graphs (decode, FULL)` and
   `Replaying aclgraph`.
4. Warmup and real-request content hashes have an empty intersection.
5. The real-request phase reports `real recompile=0`, or clearly identifies
   uncovered graph signatures.
