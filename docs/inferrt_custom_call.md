# MS-InferRT Custom Call 机制

> 本文分析 inferrt 中 `custom_call` 的详细机制与功能点。它是 inferrt 对"未内置算子"的**优雅降级(兜底)**机制。
> 相关文档:[inferrt_architecture.md](./inferrt_architecture.md)、[inferrt_runtime.md](./inferrt_runtime.md)、[inferrt_ops.md](./inferrt_ops.md)。

## 1. 先厘清两个容易混淆的概念

| 术语 | 指什么 |
| --- | --- |
| **`Op.custom_call`** | IR 里的一个算子枚举(`ops.list` 里第 25 个),是图的**兜底逃生口** |
| **Custom Operator(自定义算子)** | 用户通过 `REGISTER_CUSTOM_OP` 注册进 `CustomOpRegistry` 的 C++ 算子类,由 `Op.custom_call` 去调 |

`custom_call` 这个 IR 算子本身不做任何计算,它是一个**转发器(dispatcher)端**:负责把控制权交给真正的实现(自定义 C++ 算子 / torch 算子 / Python 函数)。这是整个系统"优雅降级"的核心。

## 2. custom_call 如何被选中:三层触发

在 Python 前端 `torch/fx_backend.py`,一个 FX 节点走向 `Op.custom_call` 有三种触发方式:

### 2.1 没有任何显式映射(最常见)

`_get_op(target)`(`fx_backend.py:2196`)在 `_OP_MAP`、`_QUALIFIED_OP_MAP`、`Op` 枚举名、`dvm` 命名空间兜底**都找不到**时,最后一行 `return Op.custom_call`。即:**任何 torch 里没收录的算子,默认走 custom_call**。

```python
# fx_backend.py:223
    return Op.custom_call
```

### 2.2 后端明确不支持(被"打回")

`_check_and_fallback_op_by_backend_support`(`fx_backend.py:2226`)对某个已映射的 op 调 pybind 的 `_ms_inferrt_ir.check_op_support`,检查 C++ 后端有没有对应 kernel。若返回非 0,打印 fallback 日志并打回 `Op.custom_call`。

```python
if int(status) != 0:
    print(f"Op {op.name} not supported: {msg}, fallback to custom_call")
    return Op.custom_call
```

### 2.3 显式别名/改写

`register_custom_call_alias(source, target, arg_rewrite)` 可把一个 custom-call 节点重定向到另一个自定义算子,并可选地改写参数。这用于把前端某个 op 名"翻译"成运行时已有的自定义算子。

## 3. IR 层节点形态

一个被 lower 成 `custom_call` 的节点,在 C++ IR 里长这样:

```text
Node[ Op.custom_call ]
  inputs[0] = Value(String "ns.op_name")   ← 算子名,必须是 "ns.op_name" 带点
  inputs[1] = 真实参数(可以是 Tensor/标量/Tuple/...)
  inputs[N] = ...
  output    = 结果的 Value
```

Python 侧 `_prepare_call_args`(`fx_backend.py:3025`)负责把 `op_name` 作为字符串 Value 节点放到第 0 个输入:

```python
else:
    op_name = op_name.replace("::", ".")
    flat_node_args = [op_name] + flat_node_args     # ← op_name 挂到最前面
```

所以 C++ 侧 `OpCustomCall::Init` 的第一个动作就是从 `inputs[0]` 读名字(`op_custom_call.cc:28-35`):

```cpp
opName_ = inputs[kInputIOpNameIndex]->ToString();
size_t pos = opName_.find(".");
if (pos == std::string::npos) { RT_GLOG(EXCEPTION) << "Invalid op name: ..."; }
```

## 4. C++ 侧的解析链:`OpCustomCall` → 三个实现路径

`OpCustomCall::Init`(`op_custom_call.cc:28-52`)是整条链的核心,它按优先级依次尝试三路:

```text
读 op 名 (inputs[0])
   │
   ▼
① CreateCustomOperator(opName)   → CustomOpRegistry 里找已注册的自定义 C++ 算子
   │ 命中?  → OpType::CustomCallOp,委托给它
   │ 未命中?
   ▼
② OpTorchCall(opName)             → 当成 torch 算子,走 torch::jit 动态 dispatch
   │ (仅在 ENABLE_TORCH_FRONT 编译时)
   │ 命中?  → OpType::TorchCallOp
   │
   ▼
③ CHECK_IF_NULL(operatorPtr_)     → 两路都不中,直接抛异常
```

```cpp
operatorPtr_ = CreateCustomOperator(opName);
SetOpType(OpType::CustomCallOp);
#ifdef ENABLE_TORCH_FRONT
  if (operatorPtr_ == nullptr) {
    operatorPtr_ = std::make_shared<OpTorchCall>(opName_);
    SetOpType(OpType::TorchCallOp);
  }
#endif
CHECK_IF_NULL(operatorPtr_);
```

`Init` 之后,`OpCustomCall` 把 `inputs[1..]` 剥离出来存进 `input_`,然后 `InferShape`/`CalcWorkspace`/`Launch` 三个方法全部**原样转发**给内部 `operatorPtr_`。所以 `OpCustomCall` 就是一个透明代理。

> 注意最后一行的 `CHECK_IF_NULL`:如果①、②都失败,直接崩溃。所以 Python 侧在第③路出现前**已经做了拦截**——见下一节。

## 5. Python 侧的拦截:`Op.custom_call` 其实是"最后再找一次"

关键区别在于:**Python 侧选择 Op 的时候已经重新检查了一次注册**,不是直接把锅甩给 C++。`_prepare_call_args`(`fx_backend.py:3025-3042`):

```python
if op == Op.custom_call:
    source_op_name = op_name
    arg_rewrite = get_custom_call_arg_rewrite(source_op_name)   # ① apply 改写 hook
    if arg_rewrite is not None:
        flat_node_args = arg_rewrite(node, flat_node_args)
    op_name = get_custom_call_alias(op_name)                    # ② 别名替换
    if op_name != source_op_name and not is_op_registered_by_custom_or_torch(op_name):
        raise RuntimeError("Custom-call alias target ... not registered")
    if not is_op_registered_by_custom_or_torch(op_name):        # ③ 兜底判定
        print("Unregistered custom/torch op: ... fallback to python_call")
        flat_node_args = _build_python_call_args(node, flat_node_args)
        op = Op.python_call                                    # ← 改成 Python 函数调用
    else:
        op_name = op_name.replace("::", ".")
        flat_node_args = [op_name] + flat_node_args
```

`is_op_registered_by_custom_or_torch`(`torch/utils.py:304`)用 pybind 的 `is_custom_op_registered`(查 `CustomOpRegistry`)配合 torch.ops 命名空间判断。所以真正决定走哪条路的是**这一层**,C++ 侧 ②的 `OpTorchCall` 主要作为"Python 侧说注册了、但自定义 registry 却找不到"的兜底。

这形成了完整的**四级降级链**:

```text
有 kernel 的内置算子(走 MrtOp)
   ↓ 没收录 / 后端不支持 / 被别名
Op.custom_call
   ↓ 自定义 registry 命中 → OpType::CustomCallOp(走 C++ 自定义算子)
   ↓ registry 未命中但 torch 认识 → OpType::TorchCallOp(走 torch::jit)
   ↓ 两边都不认识 → Op.python_call(走 Python 函数)
   ↓ 都失败 → 抛出 RuntimeError
```

## 6. 自定义算子的两种"编写/加载"方式

### 6.1 开发者写的内置自定义算子(静态/动态库)

用 `REGISTER_CUSTOM_OP` / `REGISTER_CUSTOM_OP_WITH_FACTORY` 宏,把算子类注册进 `CustomOpRegistry`(**不需要**改 `ops.list`):

```cpp
// include/custom_op_api.h 提供基类
class CustomDivOperator : public AclnnCustomOperator {
 public:
  CustomDivOperator() : AclnnCustomOperator("aclnnDiv") {}
  OpsErrorCode CalcWorkspace(...) { GetExecutor()->GetWorkspaceSize(...); return SUCCESS; }
  OpsErrorCode Launch(...) { GetExecutor()->Launch(...); return SUCCESS; }
};
REGISTER_CUSTOM_OP(custom_div, CustomDivOperator);
```

`AclnnCustomOperator`(`aclnn_custom_operator.h`)内部封装了 `AclnnExecutor`,让自定义算子无需自己管 workspace/launch 的 aclnn 细节。

### 6.2 用户现场编译加载(Python 侧)

`test_aclnn_custom.py` 展示了完整用户流程:`ms_inferrt.ops.load(name, sources, backend)` 现场用 ninja 编译 `.cc` 成 `.so`(链接 `-linferrt`),`REGISTER_CUSTOM_OP` 在 `.so` 加载时自注册。Python 用 `torch.library.custom_op("ms_inferrt::custom_div", ...)` 声明 + `register_fake` 提供 shape 元信息,`torch.compile` 时这个节点被 `_get_op` 落到 `custom_call`。

> 这就是 `ops/load.py`(`CustomOpLoader`)的作用——它生成了 ninja 构建文件、算 compile hash 缓存、用 `ctypes.CDLL` 加载。它的 `_get_ascend_environment` 里 link 的是 `ascendcl / inferrt / ops_ascend_aclnn_common`。

## 7. `OpTorchCall`:任意 torch 算子的 eager 兜底

`OpTorchCall` 是给"torch.ops 里存在、但 inferrt 没实现"的算子用的。机制:

- **Init**(`op_torch_call.cc:475`):用 `torch::jit::getAllOperatorsFor(Symbol::fromQualString("ns::op"))` 枚举 torch 侧所有 schema,逐个 `MatchOpSchema`(用实际输入类型匹配),命中就用 `op->getOperation()` 拿到可调用的 `Operation` 句柄缓存起来。匹配失败会打印很友好的错误(`GetInputTypesExpr` 列出输入类型 + 尝试过的 schema)。
- **执行时**(`CalcWorkspace`:529-542):把 `ir::Value` 输入转成 `torch::jit::Stack`(`ConvertInputsToStack` 用跳表 dispatch 各类型,含 Tensor/标量/Tuple/None),调 `operation_(stack)`,再把输出 `IValue` 转回 `ir::Value`(`ConvertStackToOutput`)。
- **`NeedLaunch() = false`**,所以 `Executor::Run` 里 `if (!opRunner.NeedLaunch()) continue;` 直接跳过 `Launch`。也就是说它**每帧在 CalcWorkspace 阶段就同步算完了**——这本质是"回落 eager 执行",不是异步 kernel 下发。`CheckOutputInputRef` 会拒绝 input/output 共享同一 storage 的情况(防止 in-place 破坏语义)。

## 8. `OpPythonCall`:直接调 Python 函数

当算子既不注册、torch 也没有时,降到 Python 函数。机制:

- 节点输入是 `[module_name, op_name, *args]`(`_build_python_call_args`:`fx_backend.py:3009`)。
- `Init` 用 pybind `importlib` 按 `module_name.op_name` 解析出 `py::function pyFunc_` 缓存。
- 执行在 `CalcWorkspace`:`PreprocessInputs` 把输入转成 Python 对象(张量借零拷贝 `at::Tensor`),`pyFunc_(*args)` 调用,`PostprocessOutputs` 把结果转回 `ir::Value`。同样 `NeedLaunch()=false`。
- 常用来兜底 Python 端纯逻辑算子(如 `setitem_impl.py` 里某些场景返回 `Op.python_call`)。

## 9. 与 `linalg_call` / `dvm_call` 的关系

这两是另外两个旁路逃生口,不是 custom_call 的子集:

- **`dvm_call` / `dvm_call_v2`**:迁移自 dvm(另一套 ML 编译后端)的图-level 调用,`fx_mlir_backend.py` 里 `MOPT_ENABLE_LINALG_CALL` 控制 lower 到 `linalg_call` 还是 `dvm_call`。它是把整个 outlined fusion region 作为一个黑盒 op。
- 它们的定位是"整块降级",而 custom_call 是"单算子降级"。两者并存,说明 inferrt 对"跑不通的算子"保留了两套不同的兜底粒度。

## 10. 功能点汇总(扩展时的落点)

| 功能点 | 位置 | 说明 |
| --- | --- | --- |
| **自定义算子注册** | `src/ops/custom_op_register.h`(`REGISTER_CUSTOM_OP`) | 运行时动态注册,不依赖 `ops.list` |
| **自定义算子 C++ 基类** | `include/custom_op_api.h` → `AclnnCustomOperator` | 封装 `AclnnExecutor`,只需实现 `CalcWorkspace`/`Launch` |
| **Python 侧加载** | `python/ms_inferrt/ops/load.py`(`CustomOpLoader`) | 现场编译 `.cc`→`.so`,文件锁 + 哈希缓存 |
| **Python 侧能否识别注册** | pybind `pybind_api.cc:27` `is_custom_op_registered` | 供 `is_op_registered_by_custom_or_torch` 查询 |
| **别名 + 参数改写** | `fx_backend.py` `_CUSTOM_CALL_ALIASES` / `_CUSTOM_CALL_ARG_REWRITES` | `register_custom_call_alias` |
| **后端支持探测** | `pybind_ir.cc:194` `check_op_support` | 决定是否把 op 打回 custom_call |
| **torch 算子兜底** | `src/ops/op_base/op_torch_call.*` | 走 `torch::jit` 动态 dispatch |
| **Python 函数兜底** | `src/ops/op_base/op_python_call.*` | 走 pybind 调 Python 回调 |
| **转发器本体** | `src/ops/op_base/op_custom_call.*` | 透明的三层转发壳 |
| **IR 枚举** | `src/ops/op_def/ops.list` 的 `OP(custom_call)` | 通用枚举,不要动它 |

## 11. 值得注意的几点

1. **`custom_call` 不参与真实 shape 推断的"自研"路径**:它把 `InferShape` 也委托给内部 `operatorPtr_`,而 `OpTorchCall`/`OpPythonCall` 的 "InferShape" 实际是**直接执行 torch/Python 函数**来得到输出(顺便带出 shape)。所以这两个兜底在 dynamic shape 模式下也能工作,但代价是**必须在 host 侧同步跑一遍**,没有 device kernel 的异步性。

2. **`OpTorchCall` 的 schema 匹配很挑剔**:要求输入类型严格匹配 torch schema。若匹配不上,`Init` 直接 `RT_GLOG(EXCEPTION)` 崩溃,所以 Python 侧通常靠 `_decompose_ops_with_fake_mode` 等 pre-pass 把算子先 decompose/replace 成 inferrt 认识的形态,而不是依赖 torch_call 兜底。

3. **`CheckOutputInputRef` 是安全阀**:custom/python/torch call 都禁止 input 与 output 共享同一 storage,因为这类 op 无法保证正确的零拷贝/in-place 语义。这跟 view 算子(有意共享 storage)形成对照。

4. **`NeedLaunch()=false` 的算子仍有完整内存生命周期**:虽然跳过 `Launch`,但 `Executor::Run` 仍会先 `AllocateMemory`/`CalcWorkspace`/`AllocateWorkspaceMemory`/`FreeMemory`,所以 workspace 内存、output 分配都正常走,只是不真正下发 device kernel。
