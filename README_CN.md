# MS-InferRT

MS-InferRT 是面向推理阶段的轻量、高性能运行时。它将 PyTorch 计算图（`torch.compile`）下沉到自有 IR 与运行时，在 Ascend / CPU 后端执行，重点优化推理时延与内存复用，并原生支持 view 类零拷贝算子。

> English version: [README.md](./README.md)

## 目录

- [快速开始](#快速开始)
- [环境依赖](#环境依赖)
- [构建](#构建)
- [安装](#安装)
- [跑通第一个例子](#跑通第一个例子)
- [运行测试](#运行测试)
- [常用环境变量](#常用环境变量)
- [文档](#文档)

## 快速开始

从源码构建到跑通一个推理示例，最短路径如下：

```bash
# 1. 构建 Ascend 后端 wheel 包（并发度 64）
bash build.sh -b ascend -j64

# 2. 安装构建产物
pip install output/*.whl

# 3. 跑一个最小示例（见“跑通第一个例子”）
python quickstart.py
```

CPU 后端把 `-b ascend` 换成 `-b cpu`（或省略，默认即 CPU）。

## 环境依赖

- GCC >= 9.0
- Python 3.9+
- Ascend 后端需安装匹配版本的 CANN 与 `torch-npu`

运行依赖（`requirements_ascend.txt` / `requirements_cpu.txt`）：

```text
torch==2.9.0
torch-npu==2.9.0   # 仅 Ascend 后端
numpy>=1.26.0
pyyaml
```

构建依赖（`requirements_build_ascend.txt`）：

```text
build
setuptools>=80.9.0
packaging>=26.0
wheel>=0.46.3
pybind11==3.0.1
nanobind>=2.9
ninja
```

完整依赖以仓库根目录对应的 `requirements_*.txt` 为准。

## 构建

典型构建命令：

```bash
bash build.sh -b ascend -j64
```

常用选项：

| 选项 | 说明 |
| --- | --- |
| `-b {cpu\|ascend}` | 选择构建后端，默认 `cpu` |
| `-jN` | 并行任务数，默认 8，越大越快但更耗资源 |
| `-f {ms\|pt}` | 限定前端编译范围（MindSpore / PyTorch），默认两者都编 |
| `-t` | 构建并运行测试（UT） |
| `-O` | 开启编译优化 |
| `-i` | 增量构建 |
| `-D` | 构建 Debug 版本（默认 Release） |
| `-d 模块列表` | 开启指定模块日志，逗号分隔，可选 `lexer,parser,compiler,vm,tensor,ops,pass,runtime,py` |
| `-e` | 从 gitee 下载 CMake 依赖并参与编译 |

完整选项见 `bash build.sh -h`。

## 安装

构建产物（whl 包）位于 `output/` 目录：

```bash
ls output/*.whl
pip install output/*.whl
```

## 跑通第一个例子

MS-InferRT 以 `torch.compile` 后端形式接入。把模型或函数用 `backend` 编译后，即按 InferRT 运行时执行：

```python
# quickstart.py
import torch
from ms_inferrt.torch import backend


def model(x, y):
    z = x + y
    z = z.view(4, -1)        # view 类算子走零拷贝路径
    return torch.relu(z)


# Ascend 后端将张量放到 npu，CPU 后端去掉 .npu() 即可
x = torch.randn(2, 8).npu()
y = torch.randn(2, 8).npu()

compiled = torch.compile(model, backend=backend)
out = compiled(x, y)
print(out.shape)
```

要点：

- `from ms_inferrt.torch import backend` 拿到编译后端
- `torch.compile(fn, backend=backend)` 之后照常调用即可
- view / reshape / permute / slice 等算子默认走 InferRT 的零拷贝 view 实现，详见 [view 特性使用说明](./docs/inferrt_view.md)

## 运行测试

### 系统测试（ST）

```bash
# 默认（Ascend 与 CPU）
bash tests/st/runtest.sh

# 仅 Ascend
bash tests/st/runtest.sh ascend

# 仅 CPU
bash tests/st/runtest.sh cpu
```

单个用例可直接用 pytest：

```bash
pytest tests/st/inferrt/ops/test_view.py -k test_view -s
```

ST 进程模型说明：

- 每个被 `@arg_mark` 标记的 `test_xxx()` 启动独立进程
- 同一用例的多个参数组合在同一进程内运行，会共享图编译缓存、环境变量等进程内资源
- 涉及多进程隔离时，用例责任人需在脚本内自行通过 `os.system` 等方式管理

### 单元测试（UT）

```bash
bash tests/ut/runtest.sh cpp
```

## 常用环境变量

| 变量 | 作用 |
| --- | --- |
| `MS_INFERRT_DISABLE_CAST_ELIMINATION` | 置为 `1` 时关闭 no-op cast 消除，该 pass 默认开启。 |
| `MS_INFERRT_DISABLE_VIEW_OPS` | 关闭指定 view 算子的零拷贝实现，回退到非 view 路径，用于排障与对比。取值为逗号分隔的算子名，或 `all`。详见 [view 特性使用说明](./docs/inferrt_view.md) |
| `MS_INFERRT_DEV_DUMP_IR` | 置为 `1` 时 dump 编译生成的 IR，便于调试 |

## 文档

- [view 特性使用说明](./docs/inferrt_view.md) — 怎么用、白名单开关、简要设计
