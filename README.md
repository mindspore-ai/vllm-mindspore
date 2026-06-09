# MS-InferRT

MS-InferRT is a lightweight, high-performance runtime for the inference phase. It lowers PyTorch graphs (via `torch.compile`) onto its own IR and runtime, executing them on Ascend / CPU backends. It focuses on inference latency and memory reuse, with native support for zero-copy view operators.

> 中文文档：[README_CN.md](./README_CN.md)

## Table of Contents

- [Quick Start](#quick-start)
- [Requirements](#requirements)
- [Build](#build)
- [Install](#install)
- [Run Your First Example](#run-your-first-example)
- [Running Tests](#running-tests)
- [Environment Variables](#environment-variables)
- [Documentation](#documentation)

## Quick Start

The shortest path from source to a running inference example:

```bash
# 1. Build the Ascend backend wheel (parallelism 64)
bash build.sh -b ascend -j64

# 2. Install the build output
pip install output/*.whl

# 3. Run a minimal example (see "Run Your First Example")
python quickstart.py
```

For the CPU backend, replace `-b ascend` with `-b cpu` (or omit it; CPU is the default).

## Requirements

- GCC >= 9.0
- Python 3.9+
- Ascend backend requires a matching CANN release and `torch-npu`

Runtime dependencies (`requirements_ascend.txt` / `requirements_cpu.txt`):

```text
torch==2.9.0
torch-npu==2.9.0   # Ascend backend only
numpy>=1.26.0
pyyaml
```

Build dependencies (`requirements_build_ascend.txt`):

```text
build
setuptools>=80.9.0
packaging>=26.0
wheel>=0.46.3
pybind11==3.0.1
nanobind>=2.9
ninja
```

Refer to the corresponding `requirements_*.txt` files in the repository root for the complete list.

## Build

Typical build command:

```bash
bash build.sh -b ascend -j64
```

Common options:

| Option | Description |
| --- | --- |
| `-b {cpu\|ascend}` | Select the build backend. Default: `cpu` |
| `-jN` | Number of parallel build jobs. Default: 8 |
| `-f {ms\|pt}` | Restrict the frontend (MindSpore / PyTorch). Default: both |
| `-t` | Build and run tests (UT) |
| `-O` | Enable compiler optimizations |
| `-i` | Incremental build |
| `-D` | Build the Debug variant (default: Release) |
| `-d <modules>` | Enable module logging, comma-separated. Options: `lexer,parser,compiler,vm,tensor,ops,pass,runtime,py` |
| `-e` | Download CMake dependencies from gitee during the build |

Run `bash build.sh -h` for the full list.

## Install

The build output (wheel package) is placed under `output/`:

```bash
ls output/*.whl
pip install output/*.whl
```

## Run Your First Example

MS-InferRT plugs in as a `torch.compile` backend. Once a model or function is compiled with `backend`, it runs on the InferRT runtime:

```python
# quickstart.py
import torch
from ms_inferrt.torch import backend


def model(x, y):
    z = x + y
    z = z.view(4, -1)        # view operators take the zero-copy path
    return torch.relu(z)


# For the Ascend backend, move tensors to npu; drop .npu() for the CPU backend
x = torch.randn(2, 8).npu()
y = torch.randn(2, 8).npu()

compiled = torch.compile(model, backend=backend)
out = compiled(x, y)
print(out.shape)
```

Key points:

- `from ms_inferrt.torch import backend` provides the compile backend.
- Call the function as usual after `torch.compile(fn, backend=backend)`.
- `view` / `reshape` / `permute` / `slice` and similar operators take the zero-copy view path by default. See the [view feature guide](./docs/inferrt_view.md).

## Running Tests

### System Tests (ST)

```bash
# Default (Ascend and CPU)
bash tests/st/runtest.sh

# Ascend only
bash tests/st/runtest.sh ascend

# CPU only
bash tests/st/runtest.sh cpu
```

Run a single case directly with pytest:

```bash
pytest tests/st/inferrt/ops/test_view.py -k test_view -s
```

ST process model:

- Each `test_xxx()` annotated with `@arg_mark` launches its own process.
- Multiple parameter combinations of the same case run in one process and share in-process resources such as the graph compilation cache and environment variables.
- When process isolation is required, the case owner manages it explicitly within the script (for example via `os.system`).

### Unit Tests (UT)

```bash
bash tests/ut/runtest.sh cpp
```

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `MS_INFERRT_DISABLE_VIEW_OPS` | Disable the zero-copy implementation of specific view operators and fall back to the non-view path, for debugging and comparison. Accepts a comma-separated list of operator names, or `all`. See the [view feature guide](./docs/inferrt_view.md) |
| `MS_INFERRT_DEV_DUMP_IR` | When set to `1`, dumps the compiled IR for debugging |

## Documentation

- [View feature guide](./docs/inferrt_view.md) — how to use it, the whitelist switch, and a brief design overview
