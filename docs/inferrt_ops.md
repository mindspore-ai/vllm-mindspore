# MS-InferRT View算子相关

## 1. Operator - 算子基类

`Operator` 是所有算子的抽象基类，定义统一的算子接口。

**核心接口**:

```cpp
class Operator {
    // 初始化
    virtual void Init(const std::vector<const ir::Value *> &input,
                   const ir::Value *output);

    // 形状推理（不同后端共享）
    virtual OpsErrorCode InferShape(const std::vector<const ir::Value *> &input,
                                ir::Value *output);

    // 工作空间计算
    virtual OpsErrorCode CalcWorkspace(const std::vector<const ir::Value *> &input,
                                  const ir::Value *output,
                                  size_t *workspaceSize);

    // 启动算子（不同后端各自实现）
    virtual OpsErrorCode Launch(const std::vector<const ir::Value *> &input,
                            void *workspace, size_t workspaceSize,
                            ir::Value *output, void *stream) = 0;

    // 就地操作支持
    virtual std::vector<std::pair<uint32_t, uint32_t>> GetOutputInputRefPairs() const;

    // 执行后形状更新
    virtual bool NeedUpdateOutputShapeAfterLaunch() const;
};
```

**错误码**:

```cpp
enum OpsErrorCode {
    SUCCESS = 0,
    INVALID_PARAM,
    INVALID_SHAPE,
    INVALID_INPUT_NUM,
    INVALID_DEVICE_ADDR,
    UNKNOWN_ERROR = 1000
};
```

**核心特性**:

- 统一的算子接口
- 形状推理和工作空间计算共享
- 不同后端各自实现 `Launch()`
- 支持就地操作（in-place operations）
- 支持执行后形状更新（如 `Unique` 算子）

## 2. 算子注册机制

算子通过宏进行注册：

```cpp
#define MRT_REG_OP(OP_NAME, OP_CLASS, DEVICE_TYPE) \
    // 注册逻辑
```

**注册示例**:

```cpp
MRT_REG_OP(all_gather, HcclAllGather, Ascend);
MRT_REG_OP(all_reduce, HcclAllReduce, Ascend);
```

## 3. 算子分类

1. **CPU 算子**: `ops/cpu/`
  - 基础算子的 CPU 实现

2. **Ascend 算子**: `ops/ascend/`
  - `aclnn/`: ACLNN 算子接口
  - `hccl/`: HCCL 通信算子
  - `lowered/`: Lowered 算子
  - `custom/`: 自定义算子
  - `dvm/`: DVM 算子
  - `mem/`: 内存算子

3. **算子基类**: `ops/op_base/`
  - 算子基类和工具函数

4. **算子定义**: `ops/op_def/`
  - 算子名称和枚举定义

## 4. OpRunner - 算子运行器

`OpRunner` 封装单个算子的运行时信息。

**核心职责**:

- 存储算子实例
- 管理输入输出张量
- 管理工作空间内存
- 处理流同步
