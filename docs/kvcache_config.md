# 参数调整示例

## 1. 背景
当前版本暂未支持profile_run功能。
根据默认设置，容易出现因为预留显存不足，导致模型无法顺利跑通的问题。当前根据过去案例给出经验值参考。

在大模型推理中，单张卡的显存占用一般分为以下几部分，以64GB显存的NPU为例:

1. 系统+驱动占用，一般为3GB左右。该部分不管是ms框架，还是CANN都无法使用
2. 非推理框架占用，一般在1GB以下。CANN占用
3. 推理框架(mindspore)占用，一般较大，和模型强相关。包含模型权重，模型运行时的激活值
4. kvcache占用

## 2. 参数介绍
1. [环境变量] `ASCEND_TOTAL_MEMORY_GB`

   默认值：`export ASCEND_TOTAL_MEMORY_GB=64`

   含义：机器中单张卡的物理显存值，单位GB。 一般为64GB, 32GB。和硬件强相关，不能随意调整， 可通过`npu-smi info`获取

2. [环境变量] `vLLM_MODEL_MEMORY_USE_GB`

   默认值：`export vLLM_MODEL_MEMORY_USE_GB=50`

   含义：机器中单张卡分配给框架占用(模型权重和模型激活值)+非框架占用的最大值， 单位GB。该值小了，模型无法跑通，出现oom。该值大了，留给kvcache空间不足，无法支持大batchsize或长序列。

3. [参数] `gpu_memory_utilization`

   默认值：`--gpu-memory-utilization=0.9`

   含义： 必须配合ASCEND_TOTAL_MEMORY_GB使用。ASCEND_TOTAL_MEMORY_GB * gpu-memory-utilization 决定留给(非推理框架+推理框架+kvcache)的最大值。

4. [参数]`max_num_seqs`

   默认值：`--max-num-seqs=256`

   含义：一次推理支持的最大batch数

5. [参数]`max_model_lens`

   默认值:` --max_model_len=16384`

   含义，一次推理中单条请求的token总长度
6. [参数]`max_num_batched_tokens`

   默认值：`--max-num-batched-tokens=16384`

   含义：一次推理能支持的最大tokens数。（>= max_num_seqs * max_model_lens）

## 3.调整思路

### 3.1 原理

模型推理中，模型权重+模型激活值需要根据实际预跑一次，从实际推理时, 框架的INFO级别日志(export GLOG_v=1)给出的大小来计算。

**实际原理为：在prefill阶段运行需要支持的最大输入长度的场景下，预跑一次。得出实际运行该场景时的模型权重+模型激活大小，计算出能分配的最大kvcache空间**

在INFO日志中，搜“actual peak used”字段，该值表明模型推理过程中，“模型权重+模型激活+kvcache大小”的实际总大小。日志样例如下，在样例中，actual peak used大小为29614 MB, 也就是28.9GB。
```
The dynamic memory pool stat info : {"in used mem":28688.13MB,"peak used mem":29331.56MB,"alloc mem":29696.00MB,"idle mem":1007.86MB,"used by event mem":0.00MB,"eager free mem":0.00MB}, actual peak used mem:29614M. Weight used size:0M, constant value used size:0M, kernel output used size:0M, other used size:28688M.
```
### 3.2 模拟预跑
#### 3.2.1 拉起预跑服务
```
以64GB显存的NPU为例：
export GLOG_v=1
export ASCEND_TOTAL_MEMORY_GB=64
export vLLM_MODEL_MEMORY_USE_GB=50
--gpu-memory-utilization=0.9
--max-num-seqs=1
--max_model_len=16384
--max-num-batched-tokens=16384
```
> 不同机器显存值不同，需要对应修改ASCEND_TOTAL_MEMORY_GB、vLLM_MODEL_MEMORY_USE_GB、gpu-memory-utilization.

> 如果机器里系统和驱动占用过大（通过`npu-smi info`查看），需要降低gpu-memory-utilization的值到0.9以下，如0.85， 0.8等;

> 如果模型参数在16k输入条件，执行模拟预跑提示oom， 需要提高vLLM_MODEL_MEMORY_USE_GB(**该值最大不超过ASCEND_TOTAL_MEMORY_GB * gpu-memory-utilization**)

> 如果还是无法跑下，说明当前环境不支持16k的长度，需要把max_model_len降低到8k， 4k等
---
#### 3.2.2 发送预跑请求+分析日志
构造16384长度的请求，执行推理。该配置表明，模型一次推理最大batch_size为1，输入的token长度为16384。
分析过程如下

1. 当前分配kvcache的大小为 64GB * 0.9 - 50GB = 7.6GB。
2. (模型权重+模型激活) = Actual_peak_used的值 - 7.6GB
3. 真实能给最大的kvcache =  64GB * 0.9 - (模型权重+模型激活)
4. 用第3步得到的最大可用kvcache空间，计算能得到的最大token数。
   > 这里获取到的kvcache值是GB的大小，需要根据模型的层数，每层kvcache单个token所有占用的字节数，综合计算能得到最大的token数。
   > 以浮点deepseekv3-671B为例：假设当前可用kvcache为8GB， dsk3模型一共num_layers层attention，单个token每层attention需要浮点的kcache大小(num_heads * head_size * 2)。
   > 最大token数 = (8 * 1024 * 1024 * 1024) / （num_layers * num_heads * head_size * 2）
5. 用第4步得到的最大token数，除max_model_len(16384)， 可得在这组参数条件下的最大并发数(max-num-seqs)


### 3.3 正式配置

正式配置：
```
export ASCEND_TOTAL_MEMORY_GB=64
export vLLM_MODEL_MEMORY_USE_GB=50 (由3.2.2中第2步计算得出)
--gpu-memory-utilization=0.9
--max-num-seqs=1 (由3.2.2中第5步计算得出，必须小于max-num-batched-tokens//max_model_len)
--max_model_len=16384 (模型支持长度16k)
--max-num-batched-tokens=16384 (必须小于3.2.2中第4步得出的最大tokens数)
```

## 4  调整总结
1. ASCEND_TOTAL_MEMORY_GB和硬件强相关(通过`npu-smi info`获取)，不建议修改
2. max_model_lens为当前模型需要支持的长度，预设条件，不建议修改
3. vLLM_MODEL_MEMORY_USE_GB为给模型实际运行的激活值大小和权重，如果模型oom，可调大改值，如51GB， 52GB等。**上限必须小于ASCEND_TOTAL_MEMORY_GB * gpu-memory-utilization， 否则kvcache的block计算会出现负值**
4. gpu-memory-utilization影响kvcache分配的大小，如果模型oom，可调小改值，如0.87, 0.85等
5. 在调整vLLM_MODEL_MEMORY_USE_GB和gpu-memory-utilization都无法跑下模型时，说明当前硬件环境并不支持预设的max_model_lens序列长度，需要降低至8k，4k等
6. 在无法降低序列长度时，建议修改并行策略、增加卡数等，以降低模型权重和模型激活在单张卡的占用