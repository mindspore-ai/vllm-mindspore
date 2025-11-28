# MS-InferRT

## 构建说明

### 环境依赖

- GCC >= 9.0
- build
- setuptools
- packaging>=24.2
- wheel>=0.34.2
- pybind11
- nanobind >= 2.9

更多依赖详见`requirements_ascend.txt`。

### 构建方式

#### 典型构建命令

```bash
bash build.sh -b ascend -j64
```

该命令以 Ascend 后端为目标（`-b ascend`），并发度设置为 64（`-j64`）以加速构建。常用可选项说明如下：

- `-b {cpu|ascend}`：选择构建目标，默认 `cpu`。
- `-jN`：控制并行任务数，默认 8，数值越大速度越快但消耗资源更多。
- `-O`：开启编译优化能力。
- `-t`：启用 UT 构建。
- `-d 模块列表`：开启指定模块的日志输出，逗号分隔，例如 `-d parser,compiler`（可选模块含 lexer/parser/compiler/vm/tensor/ops/pass/runtime/py 等）。
- `-i`：启用增量构建。
- `-D`：构建 Debug 版本（默认 Release）。
- `-f`：限定前端编译范围，可选 `ms`（MindSpore 前端）或 `pt`（PyTorch 前端），默认同时启用。
- `-e`：从 gitee 下载 CMake 依赖并参与编译。

更多可选项与说明可通过 `bash build.sh -h` 查看帮助信息。

构建输出的whl包位于`output`目录，可通过`pip install`安装。

## 测试验证

### UT

#### UT执行命令

```bash
bash tests/ut/runtest.sh cpp
```

### ST

#### 测试进程说明

- 每个`@arg_mark`标记的`def test_xxx()`函数启动独立进程
- 多个参数组合同一进程，涉及进程内资源共享：图编译缓存、环境变量等
- 用例责任人需手动通过`os.system`等方式在Python脚本中管理多进程

#### ST执行命令

##### Default (both Ascend and CPU)

```bash
bash tests/st/runtest.sh
```

##### Ascend (both Ascend and CPU)

```bash
bash tests/st/runtest.sh ascend
```

##### CPU

```bash
bash tests/st/runtest.sh cpu
```
