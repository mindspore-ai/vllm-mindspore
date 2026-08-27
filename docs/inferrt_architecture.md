# MS-InferRT 架构梳理

> 本文是 inferrt（MS-InferRT）运行时的整体功能与架构分析。用于快速上手 + 新增功能的落点定位。
> 相关文档：[inferrt_runtime.md](./inferrt_runtime.md)（执行器）、[inferrt_view.md](./inferrt_view.md)（零拷贝 view）、[inferrt_ops.md](./inferrt_ops.md)、[inferrt_logging.md](./inferrt_logging.md)。

## 1. 一句话定位

inferrt 是一个**为推理优化的轻量级 graph runtime**：以 `torch.compile` 的 FX 图为入口，把 PyTorch 图 lower 到自研的 IR（`mrt::ir`），经过优化 pass 和 kernel/executor 构建，最后在 Ascend 或 CPU 后端上执行。核心卖点是**推理延迟**和**内存复用**（零拷贝 view / Storage 复用 / 引用计数回收）。

代码位于仓库 `inferrt/` 下，命名空间统一为 `mrt`（MindSpore Runtime），由三部分组成：

- C++ 核心：`inferrt/src/`
- Python 前端：`inferrt/python/ms_inferrt/`
- pybind 绑定：`inferrt/python/pybind/`

## 2. 端到端数据流（主链路）

```text
torch.compile(fn, backend=backend)
        │
        ▼
[Python 前端]  inferrt/python/ms_inferrt/torch/
  Dynamo 产出 FX GraphModule
  → inferrt 的 backend() 逐节点 lower
  → 调用 hook 规整参数/输出
  → 不支持的 op 回退 custom_call
  （用 SymbolicShapeManager 产出带符号形状的 Value）
        │
        ▼
[图构建 API]  ir.py GraphExecutor（Python wrapper）
  begin_graph / add_*_node / add_return_node
        │
        ▼
[C++ IR]  inferrt/src/ir/
  Graph( nodes/inputs/parameters ) — Node( op, inputs[], output )
  Value  (tagged union: Tensor/Double/Int/Bool/String/Tuple/Symbol)
  Tensor (shape/dtype/stride/storageOffset)
  Storage( 底层内存 + 生命周期/复用 )
  Symbolic( 符号形状表达式 )
        │
        ▼
[优化]  inferrt/src/optimize/pass/  PassManager
  过 pass 改写 graph（如别名、常量折叠等）
        │
        ▼
[构建]  inferrt/src/runtime/builder/  Builder
  CreateOpRunners: 每个节点 → OpRunner（绑定 Operator 实例 + 设备上下文 + stream）
  UpdateRefNodeOutputValue / RecordTensorUpdatePoint / RecordStorageFreePoint
        │
        ▼
[执行器]  inferrt/src/runtime/executor/
  Executor::Run(bool isDynamic)  串行：InferShape→Alloc→CalcWorkspace→Launch
  └ 子类：Pipeline / KernelLaunchGroup / KernelCapture(AclGraph)
        │
        ▼
[算子执行]  inferrt/src/ops/
  Operator 抽象（Init/InferShape/CalcWorkspace/Launch）
  → Ascend 算子（aclnn 封装）/ CPU 算子 / 通信算子（HCCL）
```

运行时的 `compiled_callable` 在每次调用时 `update_runtime_inputs`（把 Python/torch 输入搬到 inferrt 的 input Node 上），然后 `executor.run()`。

## 3. 按层拆解

### 3.1 Python 前端最低层：`torch/fx_backend.py`（约 3200 行，最重的一块）

`backend(gm, example_inputs)` 是 `torch.compile` 的 backend 入口。核心职责是**把 FX 图翻译成 inferrt 的图**。里面有四类可插拔的 hook 机制，是扩展新功能的主要位置：

| Hook 类型 | 注册函数 | 作用 | 示例 |
| --- | --- | --- | --- |
| `pre_flatten_hook` | `register_pre_flatten_hook` | schema 匹配前调整参数 | `2+x → x+2`（标量在前交换） |
| `arg_mapping_hook` | `register_arg_mapping_hook` | 参数映射到后端算子 schema | `rms_norm(x, shape, w, eps) → [x,w,eps]` |
| `ops_mapping_hook` | `register_ops_mapping_hook` | 根据参数动态选算子 | `add→add_scalar` |
| `output_mapping_hook` | `register_output_mapping_hook` | 输出从 tuple 投影 | `argsort → tuple[1]` |

另外：

- **`_OP_MAP`**：一张超大的 `torch op → inferrt Op enum` 映射表（含 `_NPU_OP_MAP`、`_ATB_OP_MAP`）。**新增 op 支持时这层最先被填。**
- **`_check_and_fallback_op_by_backend_support`**：通过 pybind 调 C++ 的 `check_op_support`，后端不认的 op 一律落回 `Op.custom_call`。
- **view op 开关**：`MS_INFERRT_DISABLE_VIEW_OPS` 可以关掉某个 view op 的零拷贝实现，回退到非 view 版本做对比。

### 3.2 符号形状：`torch/symbolic_shape.py`

`SymbolicShapeManager` 把 torch 的 `example_value`（含 `torch.SymInt` / FakeTensor）转成 inferrt 的 `Value`，其中动态维度用 `SymbolicExpr`（`SymbolicVar`/`SymbolicConst`）表达。这是**动态 shape 支持**的基础：`Tensor` 同时持有 `shape_`（具体）和 `symbolicShape_`（符号），运行时 `EvalSymbolicShape()` 求值，`HasDynamicShape()` = `numel_ < 0`。

### 3.3 C++ IR：`src/ir`

- `graph.h`：极简的 `Node { op, inputs[], output }` + `Graph { nodes, inputs, parameters }`。
- `value.h`：`Value` 是 tagged union，能装 Tensor/标量/Tuple/Symbol；`Tuple` 承载多输出。
- `tensor.h` / `storage.h`：数据与元数据分离。`Tensor` 存 shape/dtype/stride/**storageOffset_**，并引用 `Storage`；多个 Tensor 可共享同一 `Storage`（即**零拷贝 view** 的实现基础）。`ownsStorage_`/`SetOwnsStorage` 控制生命周期。
- `symbolic/`：符号表达式。

### 3.4 优化：`src/optimize/pass`

`PassManager`（单例）跑在 graph 上，`NodePass` 做 `Match`/`Replacement` 改写节点，`DA_REGISTER_PASS` 宏注册。当前目录下只有 `pass.cc / pass.h / ud.h` 的实现骨架，说明 **pass 体系还在演进**，新增优化 pass 会加在这里。

### 3.5 硬件抽象：`src/hardware`

- `device.h`：`Device { type(CPU/NPU), index }`。
- `hardware_abstract/device_context.h`：**统一的设备交互接口** `DeviceContext` + `DeviceResManager`（内存分配/流/事件/拷贝），子类是 `cpu/` 和 `ascend/`。**新增一种后端**要实现的接口面就是这里。
- `DeviceInterface<T, Args...>`（CRTP）把 `DeviceResManager` / `KernelExecutor` 注入到 `DeviceContext` 的模板技巧。

### 3.6 算子：`src/ops`

- `operator.h`：`Operator` 抽象基类，接口为 `Init / InferShape / CalcWorkspace / Launch`；`OpType` 有 `MrtOp / CustomCallOp / TorchCallOp / PythonCallOp`。
- `op_def/ops.list`：约 163 个 `Op` 枚举，`OP(xxx)` 宏生成 enum + `ToStr`/`MatchOp`。
- `op_register.h`：**工厂注册机制**。`MRT_REG_OP(名, 类, Ascend|CPU)` 把算子类注册到 `AscendOpFactory`/`CPUOpFactory`；运行时 `CreateOperator(name, deviceType)` 按设备取。插件以 `libops_ascend.so`/`libops_cpu.so` 动态加载。
- `op_base/`：平台无关的算子（collective、reshape、shape、matmul、mul、alias、custom_call、torch_call、python_call）。
- `ops/ascend/`：按目录分 `aclnn / atb / composite / custom / dvm / hccl / lowered / mem`。绝大多数 Ascend 算子只是 thin wrapper，把参数打包后调 `aclnnXxx`。
- `ops/cpu/`：CPU 算子。
- `custom_op_register.cc` / `kernel_lib.cc`：自定义算子（可通过 `python/ms_inferrt/ops/load.py` 现场编译 C++ 成 `.so` 加载）。

### 3.7 运行时执行：`src/runtime`

`executor.h` 里的 `ExecutionMode { Base, Pipeline, GroupLaunch, AclGraph }`，由环境变量选择：

- `Base`（默认，串行一个 OpRunner 接一个）
- `Pipeline`（流水）
- `GroupLaunch`（`MS_INFERRT_KERNEL_LAUNCH_GROUP_NUM`）
- `AclGraph`（`IsAclGraphEnabled()`）

`executor.cc` 的 `BuildExecutor()` 按 mode 从 `builderCreators` 表选 Builder。

> **注意：当前源码树是裁剪版。** `executor.cc` include 了 `pipeline/pipeline_builder.h`、`kernel_launch_group/kernel_launch_group_builder.h`、`kernel_capture/kernel_capture_builder.h`，但仓库里只落了 `builder/builder.cc`（`Builder` + `Executor` 串行模式）。这些高级执行模式来自上游完整版 da runtime，本仓库是按需裁剪/迁移中的一份。文档 `inferrt_runtime.md` 描述的就是这套更完整的上层架构。

执行的核心（`Executor::Run`）：对每个 OpRunner 依次
`UpdateTensors → InferShape → AllocateMemory → CalcWorkspace → AllocateWorkspaceMemory → FreeMemory → Launch`。

内存复用关键在 builder 的 **`RecordStorageFreePoint`**：反向遍历 graph，确定每个 `Storage` 最后一个消费者，把释放点挂到该消费者的 OpRunner 上（`SetStoragesToFree`）——这样中间 tensor 的内存可以尽早归还复用，是低峰值内存的核心。

## 4. torch → inferrt 张量转换

- `torch/utils.py::from_torch` / `to_torch`：torch Tensor（FakeTensor/真 Tensor）↔ inferrt `Value`/`Tensor`，处理 device/dtype。
- `update_runtime_inputs`：执行前把真实输入搬到 graph 的 input Node（静态输入走 cache）。

## 5. vLLM 集成

- `register_inferrt_to_vllm.py`：`InferrtAdaptor(CompilerInterface)`，patch vllm 的 `backends.make_compiler`，backend 名 `"inferrt"`，内部就是调 `fx_backend.backend`。
- `external_patch/register/`：`inferrt_patch.py` 等给 vllm/vllm-ascend 打补丁。
- `external_patch/warmup/`：AI warmup——服务 ready 前用假请求预编译可能用到的图，真实请求阶段记录是否重编译。

## 6. 测试组织

- `tests/st/inferrt/ops/test_aclnn_*.py`：每个算子一个 ST 用例（Ascend aclnn）。
- `tests/st/inferrt/{runtime,hardware,distributed}/`：运行时/硬件/分布式。
- `tests/st/runtest.sh`（ST）、`tests/ut/runtest.sh cpp`（UT）。

## 7. 新增功能的落点清单

| 要加什么 | 改哪里 |
| --- | --- |
| 新算子 | ① `src/ops/op_def/ops.list` 加 `OP(name)`；② `src/ops/op_base/ascend/cpu/` 写 `Operator` 子类 + `MRT_REG_OP`；③ `torch/fx_backend.py` 的 `_OP_MAP` 加映射，如参数需规整则加 hook |
| 新优化 pass | `src/optimize/pass/` 加 `NodePass` 子类 + `DA_REGISTER_PASS` |
| 新后端（如某加速器） | `src/hardware/<dev>/device_context` + `DeviceResManager` 实现；`src/ops/<dev>/` 注册算子 |
| 新执行模式 | `executor.h` 加 `ExecutionMode` + 新增 Builder 子类 + 加进 `builderCreators` |
| 新 view/零拷贝 op | `fx_backend.py` 的 `_VIEW_OP_SWITCH_NAMES`/`_VIEW_OP_FALLBACKS` + `src/ops/` 对应算子（靠共享 `Storage`+`storageOffset` 实现） |
| 自定义 op 加载 | `ops/load.py`（现场编译）+ `custom_op_register` |
| vLLM 侧调度 | `external_patch/register/` + `register_inferrt_to_vllm.py` |

## 8. 需要注意的点

- **C++ runtime 是裁剪版**：文档和 `executor.cc` 引用了 `Pipeline/KernelLaunchGroup/KernelCapture` 三种 Builder，但源码只含 `builder/builder.cc`（串行 Base 模式）。新增功能若依赖这些高级执行模式，需先补齐缺失实现，或走 `Executor::Run` 的串行路径。
- **算子数量规模**：`ops.list` 约 163 个枚举，但真正落到本仓库的算子实现（尤其 Ascend）是逐目录维护的，新增/迁移时注意 `CMakeLists.txt` 是否把新 `*.cc` 加进构建。
- **pass 体系仍在演进**，`src/optimize/pass/` 目前骨架较薄。
