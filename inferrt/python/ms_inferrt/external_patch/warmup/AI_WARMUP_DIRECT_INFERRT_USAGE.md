# AI Warmup + Direct InferRT 使用说明

## 目标

在不修改 vLLM、vLLM-Ascend 和 `site-packages` 的前提下：

- prefill 使用 InferRT 替换 inductor 层级 backend；
- prefill 不走 piecewise 或 ACLGraph；
- decode 使用 vLLM 原生 backend，可使用 `FULL_DECODE_ONLY` ACLGraph；
- 服务 ready 前使用 AI 搜索选择假请求并预编译可能使用的图；
- 真实请求阶段记录是否发生重编译。

## 环境变量

```bash
export ASCEND_ENV=/path/to/Ascend/cann-9.0.0/set_env.sh
export PYTHON=/path/to/conda/env/bin/python
export MODEL_PATH=/path/to/model
export WARMUP_ROOT=/path/to/repo/inferrt/python/ms_inferrt/external_patch/warmup
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
```

代码不依赖开发机绝对路径。InferRT Python 包及动态库需要预先安装，或由
调用方通过标准 `PYTHONPATH`/动态库环境提供。

## 启动流程

`managed_warmup_launcher.py` 完成：

1. 使用 direct InferRT launcher 启动 vLLM；
2. 等待 `/health`；
3. 执行 AI warmup 假请求；
4. 写入 warmup 与编译报告；
5. 验证覆盖后写 ready file。

## 四卡 Qwen3.5 示例

```bash
source "${ASCEND_ENV}"
export VLLM_PLUGINS=ascend
export VLLM_TORCH_COMPILE_BACKEND=inferrt
export INFERRT_VLLM_EXTERNAL_PATCH=1
export MS_INFERRT_PATCH_WARMUP=1
export MS_INFERRT_DECODE_USE_ORIGINAL_BACKEND=1
export MS_INFERRT_COMPILE_DECODE=0
export MS_INFERRT_GRAPH_CACHE_REPORT=1
export MS_INFERRT_DEV_DUMP_IR=1

"${PYTHON}" \
  "${WARMUP_ROOT}/managed_warmup_launcher.py" \
  --ready-file "${WARMUP_ROOT}/runtime/qwen35.ready" \
  --report-dir "${WARMUP_ROOT}/runtime/qwen35_reports" \
  --warmup-json "${WARMUP_ROOT}/runtime/qwen35_warmup.json" \
  --server-log "${WARMUP_ROOT}/runtime/qwen35_server.log" \
  --warmup-lengths auto \
  --coverage-policy broad \
  --warmup-rounds 2 \
  --warmup-max-tokens 1 \
  --warmup-decode-max-tokens 1 \
  --warmup-batch-sizes 1,2,4,8 \
  --warmup-verify-policy adaptive \
  --worker-warmup-source profile_adaptive \
  --worker-warmup-profile-max-candidates 12 \
  --cache-miss-policy report \
  -- \
  serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 22000 \
  --max-num-batched-tokens 4096 \
  --tensor-parallel-size 4 \
  --compilation-config '{"backend":"inferrt"}'
```

## AI 搜索原理

这里的 AI 搜索是编译反馈驱动的自适应搜索，不依赖固定等宽桶：

1. 根据模型上限、prefill chunk、batch 和候选长度生成搜索空间；
2. 发送与真实测试数据内容和长度错开的假请求；
3. 读取 graph compile report、图签名和 IR 数量；
4. 若候选点触发新图，保留该区域并继续细化；
5. 若相邻候选复用同一图，合并区域并停止无效扫描；
6. 覆盖稳定后写 ready file，真实请求才进入服务。

因此反馈指的是假请求执行后产生的真实编译事件，而不是模型生成文本本身。

## 数据隔离

- warmup 请求和真实请求必须使用不同文件或生成种子。
- 对两组请求保存内容哈希，要求交集为 0。
- warmup 报告只统计 ready 前的编译事件。
- `real recompile` 只统计 ready 后真实请求新增的图签名。

## 验证执行路径

Prefill 应出现：

```text
Using optimized ms_inferrt.torch.backend as vLLM torch.compile backend
Configured direct InferRT torch.compile backend
backend='inferrt'
splitting_ops=[]
use_inductor_graph_partition=False
```

并应生成 `graph_rank*.txt` 或配置的 InferRT IR 输出。

Decode 应出现：

```text
cudagraph_mode=FULL_DECODE_ONLY
Capturing CUDA graphs (decode, FULL)
Replaying aclgraph
```

不应把 `FULL_DECODE_ONLY` 误判为 prefill 使用 ACLGraph。阶段标记与运行期
dispatcher 会让共享模型图在 prefill 调用 InferRT callable，在 decode 调用
原生 callable。

## 编译覆盖结果

重点查看 warmup JSON 和 report 目录中的：

- `AI warmup`：warmup 总耗时；
- `total graphs`：各 rank 总编译图数；
- `real recompile`：真实请求阶段新增图数；
- `backend_decision`：共享模型图或阶段图的后端选择原因。

理想结果是 `real recompile=0`。若不为 0，应把新增图签名对应的长度、
batch、阶段加入下一轮候选范围，而不是直接枚举 1 到最大长度。

## 常见问题

### 服务启动变慢

warmup 把首次编译成本移动到了 ready 之前。应分别记录模型加载时间、服务
health 时间、AI warmup 时间和最终 ready 时间。

### 图数量等于 TP 数

TP worker 各自编译并持有一份 rank 图。例如 TP=4 且每个 rank 一张共享
模型图时，总图数通常显示为 4。

### 输出异常

先比较相同 prompt、采样参数、tokenizer 和最大生成长度下 eager 与 InferRT
输出。乱码或明显语义错误必须先解决，不能用性能数据掩盖正确性问题。

### 切换模型

模型权重、结构、TP、dtype、chunk 或关键编译参数改变后，应重新执行
warmup；不同模型不能默认复用同一组已编译图。
