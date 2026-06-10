# MS-InferRT View执行引擎

## 1. Executor - 执行器基类

`Executor` 提供计算图执行的基本接口。

**执行模式**:

```cpp
enum ExecutionMode : size_t {
    Base = 0,        // 基础执行模式
    Pipeline = 1,     // 流水线执行模式
    GroupLaunch = 2,  // 分组启动模式
};
```

**核心方法**:

```cpp
class Executor {
    virtual void Run(bool isDynamic);
protected:
    std::shared_ptr<std::vector<OpRunner>> opRunners_;
    std::map<hardware::DeviceType, device::DeviceContext *> deviceContexts_;
};
```

## 2. Builder - 执行器构建器

`Builder` 负责构建可执行的计算图。

**构建流程**:

```text
Graph
    ↓
SetupOpRunners()
    ├── CreateOpRunners()
    ├── UpdateRefNodeOutputValue()
    └── RecordStorageFreePoint()
    ↓
BuildExecutor()
    ↓
Executor
```

**核心方法**:

- `BuildExecutor()`: 构建执行器
- `CreateOpRunners()`: 为所有节点创建 OpRunner
- `UpdateRefNodeOutputValue()`: 更新引用节点的输出值
- `RecordStorageFreePoint()`: 记录存储释放点以优化内存

## 3. GraphExecutor - 图执行器

`GraphExecutor` 提供完整的图构建、优化和执行流程。

**执行流程**:

```text
1. BeginGraph()
2. AddParameter()
3. AddOpNode()
4. AddReturnNode()
5. EndGraph()
6. OptGraph()          // 图优化
7. BuildKernels()       // 构建 Kernel
8. BuildExecutor()      // 构建执行器
9. RunGraph()          // 执行图
```

**核心方法**:

- `BeginGraph()`, `EndGraph()`: 图构建生命周期
- `AddParameter()`, `AddOpNode()`, `AddReturnNode()`: 图节点添加
- `OptGraph()`: 运行优化 Pass
- `BuildKernels()`: 为节点构建 Kernel
- `RunGraph()`: 执行计算图
- `FreeGraphOutputs()`: 释放输出内存

**动态shape支持**:

- `RunGraph(bool isDynamic)`: 支持动态shape执行

### 3.1 串行执行模式

![image.png](https://raw.atomgit.com/user-images/assets/8606433/013fbc02-d9f3-43c2-ab24-8b93782d45dd/image.png 'image.png')

### 3.2 流水执行模式

![image.png](https://raw.atomgit.com/user-images/assets/8606433/04a19f77-baee-4b41-b93d-e6eaf211714b/image.png 'image.png')

### 3.3 并行下发执行模式

```mermaid
sequenceDiagram
    participant User as 用户代码
    participant GE as GraphExecutor
    participant Builder as Builder
    participant KLG as KernelLaunchGroupExecutor
    participant MemoryCache as MemoryCache
    participant AsyncQueue as AsyncTaskQueue

    User->>GE: BeginGraph()
    User->>GE: 添加节点
    User->>GE: EndGraph()

    GE->>GE: 图优化和内核构建
    GE->>Builder: BuildExecutor()
    Builder->>KLG: 创建KernelLaunchGroupExecutor

    User->>GE: RunGraph()
    GE->>KLG: Run(isDynamic)

    KLG->>KLG: Initialize() - 初始化组启动
    KLG->>KLG: CheckInputShapeChange() - 检查输入形状变化
    KLG->>KLG: ResetTensorCacheMemory() - 重置张量缓存

    KLG->>MemoryCache: AllocateGraphCacheMemory() - 分配图缓存内存

    par 并行调度
        KLG->>KLG: ParallelDispatchKernels()
        loop 并行算子组
            KLG->>AsyncQueue: DispatchParallelLaunchKernels()
            AsyncQueue->>OpRunner: 并行执行算子组
        end
    and
        KLG->>KLG: DispatchSerialLaunchKernels()
        loop 串行算子
            KLG->>OpRunner: 串行执行算子
        end
    end

    KLG->>MemoryCache: FreeGraphCacheMemory() - 释放图缓存内存
    KLG->>GE: 返回执行结果
```
