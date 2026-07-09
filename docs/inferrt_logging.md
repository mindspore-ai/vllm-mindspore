# MS-InferRT 日志系统

InferRT 提供两套互补的日志机制，均定义在 `inferrt/src/common/logger.h` 与 `logger.cc` 中：

- **GLOG**：基于严重级别的全局日志，受全局阈值控制，用于常规运行期信息、告警与异常抛出。
- **VLOG**：基于模块的细粒度调试日志，按位掩码控制开关，关闭时流表达式不求值，适合临时定位问题。

底层由 glog 0.7.1 提供（`cmake/glog.cmake` 拉取源码编译，符号命名空间隔离为 `mrt_private`，并定义 `USE_GLOG` 宏）；未启用 glog 时自动回退到 `std::cerr` 输出，接口保持一致。

## 1. GLOG —— 严重级别日志

### 1.1 级别定义

```cpp
enum class GLogLevel : int {
  DEBUG = 0,
  INFO,
  WARNING,
  ERROR,
  CRITICAL,
  EXCEPTION = CRITICAL,  // 与 CRITICAL 同级别，但会抛出 std::runtime_error
};
```

级别由低到高，运行期通过 `IsGlogOn(level)` 判断 `level >= 全局阈值` 时才输出。`EXCEPTION` 与 `CRITICAL` 在输出后会抛 `std::runtime_error`，且对编译器保留 `[[noreturn]]` 语义。

### 1.2 使用宏

```cpp
RT_GLOG(DEBUG)     << "调试信息";
RT_GLOG(INFO)      << "常规信息，例如 op launch 前后";
RT_GLOG(WARNING)   << "告警，可继续运行";
RT_GLOG(ERROR)     << "错误，不终止进程";
RT_GLOG(EXCEPTION) << "参数/状态非法，抛异常并附带调用栈";
```

`RT_GLOG(level)` 实际展开为 `MRT_GLOG_<level>()`，并在关闭时短路为 `static_cast<void>(0)`，不构造流对象。`EXCEPTION`/`CRITICAL` 用 `for` 循环 + `[[noreturn]] ThrowNow()` 保证抛出点对编译器可见，避免 `-Werror` 下“缺少 return”误报。

### 1.3 常用场景

`inferrt/src/common/common.h` 中的检查宏与数值转换函数均已使用 `RT_GLOG(EXCEPTION)`：

```cpp
CHECK_IF_NULL(ptr);                       // ptr 为空时抛异常
CHECK_IF_FAIL(cond);                      // cond 为 false 时抛异常
CHECK_IF_FAIL_MSG(cond, "msg");           // 带自定义信息
auto n = LongToUint(value);               // 越界时抛异常
```

业务代码（`op_runner.cc`、`executor.cc` 等）中常规信息使用 `RT_GLOG(INFO)`，错误使用 `RT_GLOG(ERROR)`：

```cpp
RT_GLOG(INFO) << "Begin launch op[" << ops::ToStr(opName_) << "], inputs=" << input_;
RT_GLOG(EXCEPTION) << "Memory leak for output of operator: " << GetOpName();
```

## 2. VLOG —— 模块细粒度日志

### 2.1 设计思路

VLOG 把代码按模块分成 **8 大模块**，每个模块占 **8 个级别**，共 **64 级**，正好占满一个 `uint64_t` 掩码 `g_mrt_vlog_mask`。运行期判断仅一次移位 + 与操作，开销极低：

```cpp
inline bool IsVlogOn(VLogLevel level) noexcept {
  return ((g_mrt_vlog_mask >> static_cast<uint8_t>(level)) & 1U) != 0;
}
```

| 模块      | 级别区间 | 掩码位段            |
| --------- | -------- | ------------------- |
| RUNTIME   | 0 – 7    | `0x00000000000000ff` |
| OPS       | 8 – 15   | `0x000000000000ff00` |
| HARDWARE  | 16 – 23  | `0x0000000000ff0000` |
| COMMON    | 24 – 31  | `0x00000000ff000000` |
| CONFIG    | 32 – 39  | `0x000000ff00000000` |
| IR        | 40 – 47  | `0x0000ff0000000000` |
| OPTIMIZE  | 48 – 55  | `0x00ff000000000000` |
| PROFILER  | 56 – 63  | `0xff00000000000000` |

每个区间下再细分子模块（如 `OPS` 下有 `OPS_ACLNN`、`OPS_ATB`、`OPS_HCCL` 等）。头文件用 `static_assert` 校验各区间不超过 8 级、总级别 < 64，保证新增子模块时不会越界。

### 2.2 使用宏

```cpp
RT_VLOG(VL_RUNTIME)        << "runtime 基础日志";
RT_VLOG(VL_OPS_ACLNN)      << "aclnn 算子细节";
RT_VLOG(VL_HARDWARE_MEMORY) << "硬件内存管理细节";

// 构造调试字符串代价较高时，用 VLOG_IS_ON 包裹，关闭时完全不执行
if (RT_VLOG_IS_ON(VL_OPS_HCCL)) {
    std::string dump = BuildExpensiveDump(input_);
    RT_VLOG(VL_OPS_HCCL) << dump;
}
```

`RT_VLOG(level)` 在级别未启用时短路为 `static_cast<void>(0)`，**右侧流表达式不会被求值**，因此可放心放入仅在调试时需要的有副作用或耗时的逻辑。

### 2.3 全部 VLOG Tag

下表为 `logger.cc` 中 `DispVLogTags()` 内置的全部 tag（也可在运行期通过 `VLOG_v=63` 打印）：

| 数值 | Tag                    | 说明                       |
| ---- | ---------------------- | -------------------------- |
| 0    | VL_RUNTIME             | runtime 模块基础级别       |
| 1    | VL_RUNTIME_DETAIL      | runtime 细节               |
| 2    | VL_RUNTIME_MEMORY      | runtime 内存               |
| 3    | VL_RUNTIME_PIPELINE    | runtime 流水线             |
| 4    | VL_RUNTIME_EXECUTOR    | runtime 执行器             |
| 5    | VL_RUNTIME_BUILDER     | runtime 构建器             |
| 6    | VL_RUNTIME_CAPTURE     | runtime 图捕获             |
| 7    | VL_RUNTIME_OTHER       | runtime 其他               |
| 8    | VL_OPS                 | ops 模块基础级别           |
| 9    | VL_OPS_ACLNN           | ops aclnn                  |
| 10   | VL_OPS_ATB             | ops atb                    |
| 11   | VL_OPS_HCCL            | ops hccl                   |
| 12   | VL_OPS_DVM             | ops dvm                    |
| 13   | VL_OPS_CPU             | ops cpu                    |
| 14   | VL_OPS_CUSTOM          | ops custom                 |
| 15   | VL_OPS_OTHER           | ops 其他                   |
| 16   | VL_HARDWARE            | hardware 模块基础级别      |
| 17   | VL_HARDWARE_ASCEND     | hardware ascend            |
| 18   | VL_HARDWARE_CPU        | hardware cpu               |
| 19   | VL_HARDWARE_MEMORY     | hardware 内存              |
| 20   | VL_HARDWARE_STREAM     | hardware stream            |
| 21   | VL_HARDWARE_COLLECTIVE | hardware 集合通信          |
| 22   | VL_HARDWARE_CAPTURE    | hardware 图捕获            |
| 23   | VL_HARDWARE_OTHER      | hardware 其他              |
| 24   | VL_COMMON              | common 模块基础级别        |
| 25   | VL_COMMON_LOADER       | common 动态库加载          |
| 26   | VL_COMMON_LOGGER       | common 日志器              |
| 27   | VL_COMMON_UTILS        | common 工具                |
| 28-30 | VL_COMMON_OTHER_0/1/2  | common 预留                |
| 31   | VL_COMMON_OTHER        | common 其他                |
| 32   | VL_CONFIG              | config 模块基础级别        |
| 33   | VL_CONFIG_ASCEND       | config ascend              |
| 34   | VL_CONFIG_ACLGRAPH     | config aclgraph            |
| 35   | VL_CONFIG_OP_PRECISION | config 算子精度            |
| 36-38 | VL_CONFIG_OTHER_0/1/2  | config 预留                |
| 39   | VL_CONFIG_OTHER        | config 其他                |
| 40   | VL_IR                  | IR 模块基础级别            |
| 41   | VL_IR_GRAPH            | IR graph                   |
| 42   | VL_IR_TENSOR           | IR tensor                  |
| 43   | VL_IR_VALUE            | IR value                   |
| 44   | VL_IR_SYMBOLIC         | IR symbolic                |
| 45   | VL_IR_DTYPE            | IR dtype                   |
| 46   | VL_IR_STORAGE          | IR storage                 |
| 47   | VL_IR_OTHER            | IR 其他                    |
| 48   | VL_OPTIMIZE            | optimize 模块基础级别      |
| 49   | VL_OPTIMIZE_PASS       | optimize pass              |
| 50   | VL_OPTIMIZE_UD         | optimize UD                |
| 51-54 | VL_OPTIMIZE_OTHER_0/1/2/3 | optimize 预留           |
| 55   | VL_OPTIMIZE_OTHER      | optimize 其他              |
| 56   | VL_PROFILER            | profiler 模块基础级别      |
| 57   | VL_PROFILER_TRACE      | profiler trace             |
| 58   | VL_PROFILER_RUNTIME    | profiler runtime           |
| 59   | VL_PROFILER_OPS        | profiler ops               |
| 60   | VL_PROFILER_MEMORY     | profiler memory            |
| 61   | VL_PROFILER_OTHER_0    | profiler 预留              |
| 62   | VL_FLOW                | 流程级别 vlog              |
| 63   | VL_DISP_VLOG_TAGS      | 打印 vlog tag 使用说明     |

## 3. 环境变量

所有配置通过环境变量在 **加载 InferRT 之前** 设置（日志器通过 `std::call_once` 在全局对象 `g_log_initializer` 构造时一次性初始化，进程生命期内不再变更）。

### 3.1 GLOG 相关

| 环境变量             | 取值                          | 默认值      | 说明                                                         |
| -------------------- | ----------------------------- | ----------- | ------------------------------------------------------------ |
| `GLOG_v`             | `0`-`4`（对应 DEBUG-CRITICAL）| `2`(WARNING)| 全局严重级别阈值，低于该级别的 GLOG 不输出                   |
| `GLOG_logtostderr`   | `0` / 非 `0`                  | 非 `0`(stderr)| `0` 时改为写文件，需配合 `GLOG_log_dir`                    |
| `GLOG_log_dir`       | 目录路径                      | 空          | 日志根目录；实际写入 `<dir>/rank_<RANK_ID>/logs`（或 `rank_<OMPI_COMM_WORLD_RANK>`，或 `pid_<pid>`）|
| `GLOG_max_log_size`  | 正整数（MB）                  | `50`        | 单个日志文件最大尺寸，超限后滚动                             |
| `GLOG_logfile_mode`  | 八进制权限                    | `0640`      | 日志文件权限                                                 |
| `GLOG_stderrthreshold` | `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL` | `WARNING` | 即便写文件，仍打到 stderr 的最低级别                       |

非法值会通过 `[WARNING] MRT:` 前缀输出告警并回退到默认值，不会中断启动。

### 3.2 VLOG 相关

| 环境变量 | 取值                                   | 默认值 | 说明                                                       |
| -------- | -------------------------------------- | ------ | ---------------------------------------------------------- |
| `VLOG_v` | 逗号分隔的级别与闭区间，如 `0,2-3,8-15`| 空(全关)| 启用对应 bit；`63` 单独使用可打印全部 tag 说明              |

`VLOG_v` 解析规则：

- 单值：`VLOG_v="9"` → 启用 `VL_OPS_ACLNN`
- 闭区间：`VLOG_v="8-15"` → 启用整个 `OPS` 模块
- 混合：`VLOG_v="0,2-3,16-23"` → 同时启用 RUNTIME 部分 + 整个 HARDWARE
- `VLOG_v="63"` → 启用 `VL_DISP_VLOG_TAGS`，启动时打印 tag 全表，便于查找可用级别
- 非法 token 会输出告警并将掩码清零（全部关闭）

### 3.3 典型用法

```bash
# 只看 WARNING 及以上
export GLOG_v=2

# 全量调试：放开 GLOG 到 INFO，并启用整个 OPS + HARDWARE 模块
export GLOG_v=1
export VLOG_v="8-15,16-23"

# 不知道有哪些 tag 时，先打印 tag 表
export VLOG_v=63

# 落盘：按 rank 分目录，单文件最大 100MB
export GLOG_logtostderr=0
export GLOG_log_dir=/var/log/inferrt
export GLOG_max_log_size=100
```

## 4. 输出格式

所有日志（GLOG 与 VLOG）统一格式，仅级别前缀不同：

```text
[<LEVEL>] <TIME> InferRT [pid:<pid>, thread id:<tid> <file>:<line> <func>] <message>
```

- `<LEVEL>`：GLOG 为 `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`；VLOG 为 `V<n>`（如 `V9`）。
- `<TIME>`：`YYYY-MM-DD-HH:MM:SS.msec.usec`（毫秒.微秒，Windows 下为空）。
- `<file>`：归一化为 `inferrt/...` 相对路径。
- `<message>`：用户流式拼出的内容。

`EXCEPTION`/`CRITICAL` 输出后会在消息尾部追加 C++ 调用栈段，便于框架开发定位：

```text
----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
inferrt/src/runtime/executor/op_runner.cc(81).
```

随后抛出 `std::runtime_error`，异常 `what()` 即为上述完整消息。

## 5. 构建集成

- 启用 glog：构建时 `cmake/glog.cmake` 自动拉取 glog 0.7.1 源码（gitee 镜像）并以 `mrt_add_pkg` 编译为动态库，宏 `USE_GLOG` 由此定义；各子模块 `CMakeLists.txt` 通过 `mrt::glog` 别名链接。
- CXX11 ABI：若设置 `PYTORCH_CXX11_ABI_VERSION`，会同步传给 glog 的 `_GLIBCXX_USE_CXX11_ABI`，保证与 PyTorch ABI 一致。
- 旧的 `build.sh -d <modules>` 模块日志开关已移除，统一由 `GLOG_v` / `VLOG_v` 在运行期控制，无需重新编译。

## 6. 迁移指引（旧接口 → 新接口）

| 旧宏            | 新宏              | 备注                       |
| --------------- | ----------------- | -------------------------- |
| `LOG_OUT`       | `RT_GLOG(INFO)`      | 常规信息                    |
| `LOG_ERROR`     | `RT_GLOG(ERROR)`     | 不终止进程的错误            |
| `LOG_EXCEPTION` | `RT_GLOG(EXCEPTION)` | 抛异常 + 调用栈，保留 noreturn 语义 |
| `DEBUG_LOG_OUT_*`（编译期） | `RT_VLOG(VL_*)`（运行期） | 由编译期开关改为运行期掩码 |

新增调试日志时，优先用 `RT_VLOG(VL_<子模块>)` 而非 `RT_GLOG(INFO)`，以免污染默认输出。提交前确认 `static_assert` 未被破坏，新增子模块需在对应区间内顺位追加并同步更新 `VL_*` 别名与 `kVLogRangeDescs` / `kVLogTagDescs` 两张表。
