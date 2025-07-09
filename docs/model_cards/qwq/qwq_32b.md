# QwQ-32B Atlas 300I Duo vLLM MindSpore 推理指南
- [QwQ-32B Atlas 300I Duo vLLM MindSpore 推理指南](#qwq-32b-atlas-300i-duo-vllm-mindspore-推理指南)
  - [模型介绍](#模型介绍)
  - [运行环境准备](#运行环境准备)
    - [MindSpore](#mindspore)
    - [MindFormers](#mindformers)
    - [vLLM](#vllm)
    - [MSAdapter （安装前一定一定要卸载torch）](#msadapter-安装前一定一定要卸载torch)
    - [vllm-mindspore](#vllm-mindspore)
  - [开源镜像](#开源镜像)
    - [启动镜像命令](#启动镜像命令)
  - [下载模型权重](#下载模型权重)
    - [注意事项：](#注意事项)
  - [服务化部署](#服务化部署)
    - [1. 准备模型配置文件](#1-准备模型配置文件)
    - [2. 启动服务化](#2-启动服务化)
    - [3. 执行推理请求测试](#3-执行推理请求测试)
  - [附录](#附录)
    - [predict\_qwq\_32b.yaml](#predict_qwq_32byaml)
    - [convert\_weights.py (权重转换脚本)](#convert_weightspy-权重转换脚本)


## 模型介绍

QwQ\-32B是千问于2025年3月6日发布的人工智能大型语言模型。这是一款拥有 320 亿参数的模型，其性能可与具备 6710 亿参数（其中 370 亿被激活）的 DeepSeek\-R1 媲美。这一成果突显了将强化学习应用于经过大规模预训练的强大基础模型的有效性。

## 运行环境准备

QwQ\-32B Atlas 300I Duo推理至少需要一台4芯机器。下面将以 TP4 为例，介绍如何拉起 QwQ\-32B 服务。

当前代码仓配套如下表所示

|代码仓|代码仓地址|0.7.3配套分支|0.8.3配套分支|
|---|---|---|---|
|mindspore|[https://gitee.com/mindspore/mindspore.git](https://gitee.com/mindspore/mindspore.git)|br\_infer\_deepseek\_duo|br\_infer\_deepseek\_duo|
|mindformers|[https://gitee.com/mindspore/mindformers.git](https://gitee.com/mindspore/mindformers.git)|br\_feature\_infer\_300iduo|[https://gitee.com/highcloud3/mindformers.git](https://gitee.com/highcloud3/vllm-mindspore.git) branch: qwq\_0.8.3|
|vllm|[https://github.com/vllm\-project/vllm.git](https://github.com/vllm-project/vllm.git)|tag： v0.7.3|tag: v0.8.3|
|MSAdapter|[https://git.openi.org.cn/OpenI/MSAdapter.git](https://git.openi.org.cn/OpenI/MSAdapter.git)|commit\_id: 88ca763027b8b9b49817e8e01ac49b4d2aa1fd22|master \(commit\_id: 8a8ab6a7df58a2e9d90b781d4\)|
|vllm mindspore|[https://gitee.com/mindspore/vllm\-mindspore.git](https://gitee.com/mindspore/vllm-mindspore.git)|duo\-dev|[https://gitee.com/highcloud3/vllm\-mindspore.git](https://gitee.com/highcloud3/vllm-mindspore.git) branch: qwq\_0.8.3|

请自行准备好python3.11环境。按照上表准备好各个仓库，具体安装方法（以0.7.3 为例）：

### MindSpore

Mindspore 可以使用编译好的python包进行安装： [https://repo.mindspore.cn/public/mindspore/llm\-infer\-300duo/ascend/aarch64/mindspore\-2.7.0\-cp311\-cp311\-linux\_aarch64.whl](https://repo.mindspore.cn/public/mindspore/llm-infer-300duo/ascend/aarch64/mindspore-2.7.0-cp311-cp311-linux_aarch64.whl)

```auto
pip install ./mindspore-2.7.0-cp311-cp311-linux_aarch64.whl
```

此外，还需安装te和hccl包。它们在CANN的安装路径下

例如 

* /path/to/Milan\_C22\_20250508/ascend\-toolkit/latest/fwkacllib/lib64/te\-0.4.0\-py3\-none\-any.whl
* /path/to/Milan\_C22\_20250508/ascend\-toolkit/latest/fwkacllib/lib64/hccl\-0.1.0\-py3\-none\-any.whl




### MindFormers

```shell
git clone https://gitee.com/mindspore/mindformers.git -b br_feature_infer_300iduo
cd mindformers
bash build.sh
pip install output/*.whl
```

### vLLM

```shell
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout -b qwq
git reset --hard v0.7.3
export VLLM_TARGET_DEVICE=empty
pip install .
```

### MSAdapter （安装前一定一定要卸载torch）

```shell
git clone https://git.openi.org.cn/OpenI/MSAdapter.git
cd MSAdapter
git checkout -b qwq
git reset --hard 88ca763027b8b9b49817e8e01ac49b4d2aa1fd22
python setup.py install
```

### vllm\-mindspore

```shell
git clone https://gitee.com/mindspore/vllm-mindspore.git -b duo-dev
cd vllm_mindspore
pip install .
```


## 开源镜像

提供开源镜像供用户使用，镜像仓库地址为：hub.oepkgs.net/oedeploy/openeuler/aarch64/mindspore_300iduo:v0.7.3，可使用以下命令拉取镜像。

```shell
docker pull hub.oepkgs.net/oedeploy/openeuler/aarch64/mindspore_300iduo:v0.7.3
```

注：目前镜像基于vllm-0.7.3分支构建

### 启动镜像命令

假设您的NPU设备安装在/dev/davinci[0-3]上，并且您的NPU驱动程序安装在/usr/local/Ascend上：

```shell
docker run \
--name QWQ-32b \
--device /dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
--network host \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-it hub.oepkgs.net/oedeploy/openeuler/aarch64/mindspore_300iduo:v0.7.3 /bin/bash
```

## 下载模型权重

从 [https://huggingface.co/Qwen/QwQ\-32B/tree/main](https://huggingface.co/Qwen/QwQ-32B/tree/main) 上下载模型权重

下面以下载到`/home/ckpt/QwQ-32B` 目录为例。

在300I Duo上运行还需要执行权重转化。脚本见附录[convert\_weights.py \(权重转换脚本\)](#convert_weightspy-%E6%9D%83%E9%87%8D%E8%BD%AC%E6%8D%A2%E8%84%9A%E6%9C%AC)：

将脚本保存在本地，执行转换脚本 `python convert_weights.py --input_path /home/ckpt/QwQ-32B --output_path /home/ckpt/QwQ`  
此时 `/home/ckpt/QwQ` 目录下有转换完成的权重，需要将其他文件也拷贝到该目录下。

接着将 `/home/ckpt/QwQ/config.json` 修改中 `"torch_dtype": "float16",`

### 注意事项：

* `/home/ckpt/QwQ-32B` 可修改为自定义路径，确保该路径有足够的磁盘空间（约 62GB）。
* 下载时间可能因网络环境而异，建议在稳定的网络环境下操作。




## 服务化部署

### 1. 准备模型配置文件

可使用附录中[predict\_qwq\_32b.yaml](#predict_qwq_32byaml)文件， 将其保存到权重目录 `/home/ckpt/QwQ` 。

在 `predict_qwq_32b.yaml` 中需要对以下配置进行修改（若为默认路径则无需修改）：

```yaml
processor:
  tokenizer:
    vocab_file: "/home/ckpt/QwQ/vocab.json"       # 配置为词表文件的绝对路径
    merges_file: "/home/ckpt/QwQ/merges.txt"      # 配置为词表文件的绝对路径
```

### 2. 启动服务化

运行下面的脚本即可

```shell
export vLLM_MODEL_BACKEND=MindFormers
export vLLM_MODEL_MEMORY_USE_GB=30
export ASCEND_TOTAL_MEMORY_GB=40
export MS_ENABLE_LCCL=off
export HCCL_OP_EXPANSION_MODE=AIV
export DEVICE_NUM_PER_NODE=8
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

export HCCL_OP_EXPANSION_MODE="AI_CPU"

# 日志相关
export GLOG_v=3
# export GLOG_logtostderr=0
# export GLOG_log_dir=/home/yyd/code/mindformers/vllm_output/glog


# 开启 profile
# export VLLM_TORCH_PROFILER_DIR=/home/yyd/code/mindformers/vllm_profile

# 并行下发
# export EXPERIMENTAL_KERNEL_LAUNCH_GROUP="thread_num:2,kernel_group_num:8"

# 保存ir
# export MS_DEV_SAVE_GRAPHS=1
# export MS_DEV_SAVE_GRAPHS_PATH=/home/yyd/code/mindformers/vllm_output/graph

# 300I Duo 新增
export MS_ENABLE_INTERNAL_BOOST=off
export MS_ENABLE_TRACE_MEMORY=off

export MS_NODE_TIMEOUT=300
export MS_ALLOC_CONF=enable_vmm:true

# 设置为predict_qwq_32b.yaml的绝对路径
export MINDFORMERS_MODEL_CONFIG=/home/ckpt/QwQ/predict_qwq_32b.yaml

# --model 需要设置为权重的绝对路径
python3 -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server --model "/home/ckpt/QwQ" --trust-remote-code --tensor-parallel-size=4 --max-num-seqs=256 --block-size=128 --gpu-memory-utilization=0.7 --max-num-batched-tokens=16384 --max-model-len=8192 2>&1 | tee log.txt
```

参数含义可参考 [vLLM参数](https://wiki.huawei.com/domains/77253/wiki/192114/WIKI202501145732773)

### 3. 执行推理请求测试

执行以下命令发起流式推理请求：  
（model项应该设置为和启动服务时的 model 项一致）

```cpp
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/ckpt/QwQ",
    "prompt": "Mindspore is",
    "max_tokens": 120,
    "temperature": 0
  }'
```



## 附录

### predict\_qwq\_32b.yaml

```yaml
seed: 0
output_dir: './output' # path to save checkpoint/strategy
load_checkpoint: ""
src_strategy_path_or_dir: ''
auto_trans_ckpt: False  # If true, auto transform load_checkpoint to load in distributed model
only_save_strategy: False
resume_training: False
use_parallel: True
run_mode: 'predict'
load_ckpt_format: 'safetensors'

# trainer config
trainer:
  type: CausalLanguageModelingTrainer
  model_name: 'qwq_32b'

# runner config
runner_config:
  epochs: 5
  batch_size: 1
  sink_mode: True
  sink_size: 2
runner_wrapper:
  type: MFTrainOneStepCell
  scale_sense:
    type: DynamicLossScaleUpdateCell
    loss_scale_value: 65536
    scale_factor: 2
    scale_window: 1000
  use_clip_grad: True

# default parallel of device num = 8 for Atlas 800T A2
parallel_config:
  data_parallel: 1
  model_parallel: 4
  pipeline_stage: 1
  micro_batch_num: 1
  vocab_emb_dp: False
  gradient_aggregation_group: 4
  use_seq_parallel: False
# when model parallel is greater than 1, we can set micro_batch_interleave_num=2, that may accelerate the train process.
micro_batch_interleave_num: 1

model:
  model_config:
    type: LlamaConfig
    batch_size: 1
    seq_length: 32768
    hidden_size: 5120
    num_layers: 64
    num_heads: 40
    n_kv_heads: 8
    vocab_size: 152064
    intermediate_size: 27648
    max_position_embeddings: 32768
    qkv_has_bias: True
    rms_norm_eps: 1.0e-6
    theta: 1000000.0
    emb_dropout_prob: 0.0
    eos_token_id: [151645,151643]
    pad_token_id: 151643
    bos_token_id: 151643
    compute_dtype: "float16"
    layernorm_compute_type: "float16"
    softmax_compute_type: "float16"
    rotary_dtype: "float16"
    param_init_type: "float16"
    use_past: True
    use_flash_attention: True
    block_size: 32
    num_blocks: 1024
    use_past_shard: False
    offset: 0
    checkpoint_name_or_path: ""
    repetition_penalty: 1.05
    temperature: 0.7
    max_decode_length: 64
    top_k: 20
    top_p: 0.8
    do_sample: False
    is_dynamic: True
    qkv_concat: True
    ffn_concat: True
    auto_map:
      AutoTokenizer: [qwen2_5_tokenizer.Qwen2Tokenizer, null]
  arch:
    type: ParallelQwenForCausalLM

processor:
  return_tensors: ms
  tokenizer:
    model_max_length: 131072
    vocab_file: "/home/ckpt/QwQ/vocab.json"
    merges_file: "/home/ckpt/QwQ/merges.txt"
    unk_token: null
    pad_token: "<|endoftext|>"
    eos_token: "<|im_end|>"
    bos_token: null
    chat_template: "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"
    type: Qwen2Tokenizer
  type: Qwen2Processor

# mindspore context init config
context:
  mode: 1 #0--Graph Mode; 1--Pynative Mode
  device_target: "Ascend"
  ascend_config:
    precision_mode: "must_keep_origin_dtype"
  max_call_depth: 10000
  max_device_memory: "39GB"
  save_graphs: False
  save_graphs_path: "./graph"
  device_id: 0
  affinity_cpu_list: None

# parallel context config
parallel:
  parallel_mode: "STAND_ALONE" # 0-data parallel, 1-semi-auto parallel, 2-auto parallel, 3-hybrid parallel
  gradients_mean: False
  enable_alltoall: False
  full_batch: True
  search_mode: "sharding_propagation"
  enable_parallel_optimizer: False
  strategy_ckpt_save_file: "./ckpt_strategy.ckpt"
  parallel_optimizer_config:
    gradient_accumulation_shard: False
    parallel_optimizer_threshold: 64

```

### convert\_weights.py \(权重转换脚本\)

```python
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
transform huggingface model to mindspore ckpt.
"""
import numpy as np
import argparse
import json
import os
from collections import defaultdict
from glob import glob
import warnings
import torch
from safetensors.torch import load_file

import mindspore as ms
from pathlib import Path

dtype_map = {
    'fp32': ms.float32,
    'bf16': ms.bfloat16,
    'fp16': ms.float16
}

def process_file(my_path):
    target_dir = Path(my_path)
    path_list = []
    if target_dir.is_dir():
        for file_path in target_dir.iterdir():
            if file_path.is_file():
                path_list.append(str(file_path))
    elif target_dir.is_file():
        path_list.append(my_path)
    return path_list

def is_skip(file, path2):
    is_skip = False
    file_name = file.split('/')[-1]
    for name in path2:
        name = name.split('/')[-1]
        if name == file_name:
            print(name)
            is_skip = True
            break
    return is_skip


def convert_ms_to_gmm(input_path, output_path):
    """convert ms routing ffn weight for gmm."""
    ckpt_path1 = process_file(input_path)
    ckpt_path2 = process_file(output_path)
    print(ckpt_path1)
    print(ckpt_path2)
    for i in range(len(ckpt_path1)):
        if not ckpt_path1[i].endswith('safetensors') or is_skip(ckpt_path1[i], ckpt_path2):
            continue
        name = ckpt_path1[i]
        params = ms.load_checkpoint(name, format="safetensors")
        for k, v in params.items():
            if 'norm'  in k:
                print(k)
                orign_tensor = ms.Tensor(v.asnumpy(), dtype=ms.float16)
                params[k] = ms.Parameter(orign_tensor)
                print(params[k])
            elif v.dtype == ms.bfloat16:
                print(k)
                orign_tensor = ms.Tensor(v.asnumpy(), dtype=ms.float16)
                params[k] = ms.Parameter(orign_tensor)
                print(params[k])
        # name = os.path.realpath(name)
        path = name.split('/')[-1]
        save_path = os.path.join(output_path, path)
        ms.save_checkpoint(params, save_path, format="safetensors")
    print(f"\rConvertion finished, the mindspore ckpt is saved in '{output_path}'.", flush=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default=None, type=str)
    parser.add_argument('--input_path', default=None, type=str)

    args = parser.parse_args()

    convert_ms_to_gmm(input_path=args.input_path, output_path=args.output_path)

```
