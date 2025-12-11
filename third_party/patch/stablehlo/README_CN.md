# StableHLO 补丁说明

本目录包含应用于 StableHLO 代码库的补丁，用于修复不同编译器版本和工具链的编译问题。

## 补丁列表

### 001-fix-compiler-warning-flags.patch

**描述：**
修复使用非 Clang 编译器（如 GCC）编译时的错误。原代码使用了 Clang 特定的警告标志，GCC 不支持这些标志，导致编译失败。

**问题：**
`CMakeLists.txt` 文件直接添加了 Clang 特定的警告标志（`-Wno-error=cast-qual`, `-Wno-error=nested-anon-types` 等），但没有检查编译器是否支持这些标志。当使用 GCC 或其他不支持这些标志的编译器时，会导致编译错误。

**解决方案：**

- 使用 CMake 的 `CheckCXXCompilerFlag` 来检测编译器对每个警告标志的支持
- 只添加支持的标志，避免编译错误
- 使构建系统在不同编译器间更加可移植

**修改的文件：**

- `stablehlo/integrations/python/CMakeLists.txt`

### 002-fix-compile-error-with-old-gcc.patch

**描述：**
修复在较旧版本的 GCC 编译器编译 `StablehloAggressiveFolder.cpp` 时出现的模板实例化错误。

**问题：**
在 `FoldUnaryOpPattern` 模板类中，嵌套的 `FoldDispatch` 结构体直接调用父类的静态模板方法 `FoldIfImplemented`。某些较旧版本的 GCC（以及某些编译器配置）无法正确实例化这些模板，导致如下错误：

```text
error: no matching function for call to
'FoldUnaryOpPattern<Impl, OpType>::FoldIfImplemented<llvm::APInt>(llvm::APInt&)'

error: incomplete type 'DirectFolderExists<Impl, llvm::APInt>' used in
nested name specifier
```

**根本原因：**
即使使用了 `if constexpr` 检查，某些编译器版本在模板解析时仍会尝试实例化所有可能的模板重载，这需要实例化 `DirectFolderExists` trait。在嵌套结构中，这可能导致模板实例化时出现不完整类型错误。

**解决方案：**

- 在 `FoldDispatch` 中引入两个辅助模板方法：
    - `foldDirect`: 当 `DirectFolderExists` trait 为 true 时，直接调用 `Impl::EvaluateOp`
    - `foldConverting`: 当 `ConvertingFolderExists` trait 为 true 时，通过转换函数调用 `Impl::EvaluateOp`
- 这些辅助方法避免了从嵌套结构体中调用 `FoldIfImplemented`，防止了模板实例化问题
- 使用 `if constexpr` 确保只编译正确的代码路径

**修改的文件：**

- `stablehlo/transforms/optimization/StablehloAggressiveFolder.cpp`

**技术细节：**

修复的工作原理：

1. 将模板实例化逻辑移到 `FoldDispatch` 内部的辅助方法中
2. 避免从嵌套结构体中直接调用父类模板方法
3. 使用编译时类型特征检查（`if constexpr`）在编译时选择正确的实现

### 003-build-remove-tests.patch

**描述：**
移除 StableHLO 测试相关的构建目标和子目录，简化构建过程。

**问题：**

1. 在将 StableHLO 作为依赖项集成到其他项目时，不需要构建和运行 StableHLO 的测试套件

**解决方案：**

- 移除根 `CMakeLists.txt` 中的测试目标定义（`check-stablehlo-ci`, `check-stablehlo-quick`, `check-stablehlo-slow`）
- 移除 `add_subdirectory(examples)` 调用，不构建示例代码
- 移除 `add_subdirectory(tools)` 调用，不构建工具代码
- 移除各级cmake文件中对test目录的构建
- 移除 `CheckCAPI` 的定义（因为依赖 `CheckOps`，而 `CheckOps` 在 tests 中构建），同时移除 `CheckPythonSources` 和 `CheckPythonExtensions` 的定义，因为他们依赖 `CheckCAPI`
- 从 `StablehloUnifiedPythonCAPI` 和 `StablehloUnifiedPythonModules` 中移除 Check 相关的依赖

**修改的文件：**

- `CMakeLists.txt`
- `stablehlo/CMakeLists.txt`
- `stablehlo/conversions/linalg/CMakeLists.txt`
- `stablehlo/conversions/tosa/CMakeLists.txt`
- `stablehlo/integrations/c/CMakeLists.txt`
- `stablehlo/integrations/cpp/builder/CMakeLists.txt`
- `stablehlo/integrations/python/CMakeLists.txt`

**技术细节：**

Check dialect 是 StableHLO 中用于测试的 dialect，其实现（`CheckOps`）和 Python 绑定（`CheckPythonExtensions`）都属于测试基础设施。由于这些组件依赖于在 tests 目录中构建的 `CheckOps` 库，而 tests 目录已被移除，因此这些组件也必须被移除。

## 应用顺序

这些补丁应按数字顺序应用（001 → 002 → ...）。
