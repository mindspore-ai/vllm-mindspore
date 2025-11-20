# Torch-MLIR 补丁说明

本目录包含应用于 Torch-MLIR 代码库的补丁，用于适配 ms_inferrt 项目的构建需求。

## 补丁列表

### 001-build-isolate-symbols.patch

**描述：**
为 TorchMLIR Python 绑定添加符号隔离，避免与其他 MLIR Python 绑定的符号冲突。

**问题：**
当 Torch-MLIR 作为其他项目的一部分构建时，其 Python 绑定可能与其他 MLIR Python 绑定（如来自 LLVM/MLIR 主项目的绑定）产生符号冲突，导致运行时链接错误或符号解析问题。

**解决方案：**

- 在 Linux 平台上为 `TorchMLIRAggregateCAPI` 目标添加链接器选项 `--exclude-libs,ALL`
- 该选项会排除库中的所有符号，避免与其他库的符号冲突
- 确保 Torch-MLIR 的 Python 绑定可以与其他 MLIR 组件共存

**修改的文件：**

- `python/CMakeLists.txt`

### 002-build-external-stablehlo.patch

**描述：**
允许外部指定 StableHLO 源码目录，而不是硬编码使用 `externals/stablehlo`。

**问题：**
Torch-MLIR 默认从 `externals/stablehlo` 目录构建 StableHLO。但在某些构建场景下，我们希望使用外部指定的 StableHLO 源码目录，例如：

- 使用独立的 StableHLO 源码树
- 在构建系统中统一管理依赖项
- 支持多个项目共享同一个 StableHLO 源码

**解决方案：**

- 引入 `USE_STABLEHLO_SOURCE_DIR` CMake 变量
- 如果未指定，则使用默认的 `externals/stablehlo` 目录
- 如果指定了，则使用指定的目录
- 同时更新 `include_directories` 以使用正确的路径

**修改的文件：**

- `CMakeLists.txt`

### 003-build-remove-tests.patch

**描述：**
移除测试相关的构建目标和子目录，简化构建过程。

**问题：**
在将 Torch-MLIR 作为依赖项集成到其他项目时，通常不需要构建和运行 Torch-MLIR 的测试套件。这些测试会增加构建时间，并且可能引入额外的依赖项。

**解决方案：**

- 移除 `check-torch-mlir-all` 自定义目标及其依赖关系
- 移除 `add_subdirectory(test)` 调用，不构建测试目录
- 移除各个子项目中的测试相关依赖
- 保留核心功能构建，减少构建时间和复杂度

**修改的文件：**

- `CMakeLists.txt`
- `projects/pt1/CMakeLists.txt`
- `projects/pt1/python/CMakeLists.txt`
- `projects/pt1/python/test/CMakeLists.txt`

### 004-build-embedded.patch

**描述：**
添加嵌入式构建模式支持，允许 Torch-MLIR 作为其他项目的嵌入式组件构建。

**问题：**
当 Torch-MLIR 作为其他项目的一部分嵌入构建时，某些初始化代码（如 Python site 初始化）可能不适用或会导致冲突。需要一种方式来区分独立构建和嵌入式构建。

**解决方案：**

- 添加 `TORCH_MLIR_BUILD_EMBEDDED` CMake 选项
- 当启用该选项时，跳过 `TorchMLIRSiteInitialize` 组件的声明和包含
- 在构建逻辑中添加第三个分支来处理嵌入式构建场景
- 确保嵌入式构建时不会执行不必要的初始化操作

**修改的文件：**

- `CMakeLists.txt`
- `python/CMakeLists.txt`

## 应用顺序

这些补丁应按数字顺序应用（001 → 002 → ...）。
