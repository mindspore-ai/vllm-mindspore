# MS-InferRT View 特性

## 1. view 特性介绍

PyTorch 中 `view`、`reshape`、`permute`、`slice` 等算子并不复制数据，而是与原张量共享同一块底层 storage，只改变对数据的逻辑解释（shape / strides / storage offset）。MS-InferRT 把这类算子原生实现为零拷贝的 view 算子：

- 不分配新内存，不下发 kernel
- 输出与输入共享同一 storage
- 只更新输出张量的 shape / strides / storage offset / format 等元数据

这样可以减少图节点数量、消除多余拷贝，并保持与 PyTorch eager 一致的 shape / stride / offset 语义。

## 2. 如何使用

view 特性是默认开启的，无需任何额外配置。把模型用 InferRT 后端编译后，命中的 view 类算子会自动走零拷贝路径：

```python
import torch
from ms_inferrt.torch import backend


def model(x):
    y = x.permute(1, 0)      # permute_view，零拷贝
    z = y.reshape(-1)        # view，零拷贝
    return z + 1


x = torch.randn(8, 4).npu()
compiled = torch.compile(model, backend=backend)
out = compiled(x)
```

下表是前端默认映射到 view 实现的常见算子：

| PyTorch 写法 | InferRT view 算子 |
| --- | --- |
| `view` / `reshape` | `view` |
| `permute` / `transpose` / `t` / `movedim` | `permute_view` |
| `flatten` | `flatten_view` |
| `slice`（`x[a:b]`） | `slice_view` |
| `select`（`x[i]`） | `select_view` |
| `squeeze` | `squeeze_view` |
| `unsqueeze` | `unsqueeze_view` |
| `narrow` | `narrow_view` |
| `unbind` | `unbind_view` |
| `split` / `chunk` / `split_with_sizes` | `split_with_size_view` |
| `split`（按固定块大小） | `split_tensor_view` |

> 这些算子既支持函数式写法（`torch.permute(x, ...)`）也支持方法式写法（`x.permute(...)`），还覆盖了对应的 `aten` 重载。

### 2.1 需要连续布局时用 contiguous

view 输出可能是非连续张量。如果下游算子要求连续内存，需要显式调用 `contiguous` 物化为一块新的连续 storage：

```python
y = x.permute(1, 0)        # 非连续 view
z = y.contiguous()         # 物化为连续张量，分配新 storage
```

`contiguous` 不改变数值，只改变内存布局，是非连续张量进入“只支持连续输入”算子前的标准落点。

## 3. 白名单开关：MS_INFERRT_DISABLE_VIEW_OPS

为了便于排障与对比，MS-InferRT 提供环境变量 `MS_INFERRT_DISABLE_VIEW_OPS`，用于按算子粒度关闭 view 实现，回退到非 view 路径。

### 3.1 用法

取值为逗号分隔的算子名（大小写不敏感，自动去空格）：

```bash
# 关闭 permute 的 view 实现
export MS_INFERRT_DISABLE_VIEW_OPS=permute

# 同时关闭 transpose 和 chunk
export MS_INFERRT_DISABLE_VIEW_OPS=transpose,chunk

# 关闭全部 view 算子
export MS_INFERRT_DISABLE_VIEW_OPS=all
```

可用的开关名（token）：

| token | 影响的算子 |
| --- | --- |
| `view` / `reshape` | `view` |
| `permute` | `permute`（方法 / 函数 / aten） |
| `transpose` | `transpose` |
| `t` | `t` |
| `movedim` | `movedim` |
| `flatten` | `flatten_view` |
| `slice` | `slice_view` |
| `select` | `select_view` |
| `squeeze` | `squeeze_view` |
| `unsqueeze` | `unsqueeze_view` |
| `narrow` | `narrow_view` |
| `unbind` | `unbind_view` |
| `split` / `split_with_size` | `split_with_size_view` |
| `split` / `split_tensor` | `split_tensor_view` |
| `chunk` | chunk 下沉的 split view |
| `all` | 全部 view 算子 |

### 3.2 回退行为

只有部分 view 算子存在等价的非 view（实算）实现，关闭时才会真正回退：

| view 算子 | 回退到 |
| --- | --- |
| `permute_view` | `permute` |
| `split_with_size_view` | `split_with_size` |
| `split_tensor_view` | `split_tensor` |

对于没有非 view 实现的算子（如 `view`、`slice_view`、`select_view`、`squeeze_view` 等），即使在开关中指定，也无法回退，会保持 view 实现并打印一条提示，例如：

```text
Disabling InferRT view op Op.view for target reshape is ignored:
no non-view implementation is registered, continue using view implementation.
```

这意味着该开关主要用于定位“某个具体 view 算子是否引入问题”，而不是关闭所有零拷贝能力。

## 4. 简要设计

### 4.1 端到端链路

```text
PyTorch FX Graph
    │  前端映射：reshape/permute/slice/... → InferRT view 算子
    ▼
InferRT IR（Tensor 携带 shape / strides / storage_offset / storageShape）
    │  view 算子标记 refPairs：输出引用哪个输入的 storage
    ▼
Runtime Builder / OpRunner
    │  按 refPairs 让输出复用输入 storage，并置 ownsStorage=false
    ▼
view 算子执行
    │  只在 CalcWorkspace 阶段更新元数据，NeedLaunch=false，不下发 kernel
    ▼
输出（与输入共享 storage 的视图张量）
```

### 4.2 三个关键机制

1. 元数据复用上游推导：view 算子不重新计算输出 shape，直接复用 torch infer shape 链路已经推好的结果，只按 view 语义计算 strides 和 storage offset，避免与 PyTorch 在边界场景（符号 shape、0 element 等）产生不一致。

2. storage 共享与生命周期：view 输出通过 `refPairs`（如单输出 `[(0,0)]`、Tuple 输出 `[(0,0),(1,0),...]`）声明引用关系；Runtime 让输出复用输入 storage，并设置 `ownsStorage=false`，确保共享 storage 不会被某个视图输出错误地 resize。

3. 不下发 kernel：view 算子的 `NeedLaunch()` 恒为 `false`，`Launch()` 直接返回成功，开销只来自图构建与元数据更新。

### 4.3 与 contiguous 的关系

view 特性负责“共享 storage 的逻辑解释”，不负责自动消除非连续布局。当算子边界要求连续内存时，由独立的 `contiguous` 算子物化到新 storage。`contiguous` 输出 `ownsStorage=true`，是 view 能力的配套出口，而非替代方案。

### 4.4 view 算子家族语义

| 算子 | 语义 |
| --- | --- |
| `view` | 在连续输入基础上重解释形状（含 `-1` 维推导） |
| `permute_view` | 维度置换，只调整 strides 顺序 |
| `slice_view` | 指定维度切片，更新该维长度、stride（乘 `step`）与 storage offset |
| `select_view` | 指定维度选点并消除该维 |
| `squeeze_view` | 删除 size 为 1 的维度 |
| `unsqueeze_view` | 在指定位置插入 size 为 1 的新维度 |
| `narrow_view` | 在某维上截取连续区间（step 为 1 的 slice） |
| `unbind_view` | 沿某维拆分并消除该维，返回多个 view 组成的 tuple |
| `split_tensor_view` | 按固定块大小切分，Tuple 输出共享同一 storage |
| `split_with_size_view` | 按给定大小列表切分，Tuple 输出共享同一 storage |
| `chunk_view` | 按 chunk 规则分块，Tuple 输出共享同一 storage |

这些算子统一通过 `AclnnViewBase` 基类落地：`Init()` 生成 `refPairs`，`NeedLaunch()` 恒为 `false`，`Launch()` 直接返回成功，输出元数据统一经由 `UpdateTensorViewInfo()` 收敛更新，保证 storageShape 继承、storage offset 计算、Tuple 与单输出行为一致。

### 4.5 典型算子的元数据推导

view 算子不重算输出 shape（复用 torch infer shape 结果），只按下表规则计算 strides 与 storage offset。

| 算子 | strides | storage offset |
| --- | --- | --- |
| `squeeze` | 删除被 squeeze 维（size=1）对应的 stride，其余保留 | 不变 |
| `unsqueeze` | 在 `dim` 处插入新 stride（`dim>=rank` 时为 1，否则 `shape[dim]*strides[dim]`） | 不变 |
| `narrow` | 继承输入 strides | `offset + start * strides[dim]` |
| `slice` | 目标维 stride 乘 `step` | `offset + start * strides[dim]` |
| `select` | 删除被选中维对应的 stride | `offset + index * strides[dim]` |
| `unbind` | 删除 `dim` 后的输入 strides | 第 `i` 个输出为 `offset + i * strides[dim]` |

示例（`narrow`）：

```text
input.shape   = [4, 5, 6]
input.strides = [30, 6, 1]
input.offset  = 0

narrow(dim=1, start=1, length=3)

output.shape   = [4, 3, 6]      # 来自 torch infer shape
output.strides = [30, 6, 1]     # 继承
output.offset  = 0 + 1 * 6 = 6
```

变长输出（`split` / `chunk` / `unbind`）通过 `refPairs` 让所有输出引用同一输入 storage（`[(0,0), (1,0), ..., (n-1,0)]`），每个输出再各自计算 shape 与 offset。相比分解为多个 `slice + squeeze`，单个多输出 view op 节点更少、元数据更集中，便于校验引用关系。

### 4.6 风险点与规避

| 风险 | 描述 | 规避方式 |
| --- | --- | --- |
| strides 被默认连续布局覆盖 | 在错误阶段重算 stride 会破坏 view 语义 | 构造、`Resize()`、shape 更新阶段不无条件重算 strides |
| Ref 输出错误 resize | 共享 storage 的 view 输出若保有 owner 权限，会破坏输入 storage | `ownsStorage=false` + runtime 统一绑定 |
| Tuple 输出绑定错误 | 变长输出未正确生成 `refPairs` 会错绑或空绑 | 由统一接口生成 `refPairs` |
| Custom Call 混入引用输出 | 通用桥接层无法安全处理引用输出 | 对 Custom Call 保持非引用输出边界 |

## 5. 常见问题

**Q: 编译后 view 的输出 stride / offset 和 eager 不一致？**
先用 `MS_INFERRT_DISABLE_VIEW_OPS` 关闭对应算子做对比，缩小到具体算子；再结合 `MS_INFERRT_DEV_DUMP_IR=1` dump IR 检查该算子是否走了预期的 view 路径。

**Q: 设了开关但算子还是走 view？**
该算子没有非 view 实现，无法回退（见 [3.2 回退行为](#32-回退行为)），此时会打印提示并继续用 view 实现。

**Q: 下游算子报“需要连续输入”之类的错误？**
在该算子前显式插入 `.contiguous()`，把非连续 view 物化为连续张量。
