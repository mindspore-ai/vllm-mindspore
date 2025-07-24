# Qwen3-32B vLLM+MindSpore 推理指南

## 目录
- [模型介绍](#模型介绍)
- [快速开始](#快速开始)
- [声明](#声明)

## 模型介绍

### 下载链接

|  社区  | 下载地址                                                      |
|:----:|:----------------------------------------------------------|
| 魔乐社区 | https://modelers.cn/models/MindSpore-Lab/Qwen3-32B |

## 快速开始
Qwen3-32B推理至少需要1台（2卡）Atlas 800T A2（64G）服务器服务器（基于BF16权重）。昇思MindSpore提供了Qwen3-32B推理可用的Docker容器镜像，供开发者快速体验。

### 下载模型权重

执行以下命令为自定义下载路径 `/home/work/Qwen3-32B` 添加白名单：

```bash
export HUB_WHITE_LIST_PATHS=/home/work/Qwen3-32B
```

执行以下 Python 脚本从魔乐社区下载昇思 MindSpore 版本的 Qwen3-32B 文件至指定路径 `/home/work/Qwen3-32B` 。下载的文件包含模型代码、权重、分词模型和示例代码，占用约 62GB 的磁盘空间：

```python
from openmind_hub import snapshot_download

snapshot_download(
    repo_id="MindSpore-Lab/Qwen3-32B",
    local_dir="/home/work/Qwen3-32B",
    local_dir_use_symlinks=False
)
```

下载完成的 `/home/work/Qwen3-32B` 文件夹目录结构如下：

```text
Qwen3-32B
  ├── vocab.json                                # 词表文件
  ├── merges.txt                                # 词表文件
  ├── predict_qwen3_32b.yaml                    # 模型yaml配置文件
  └── weights
        ├── model-xxxxx-of-xxxxx.safetensors    # 模型权重文件
        └── model.safetensors.index.json        # 模型权重映射文件
```

#### 注意事项：

- `/home/work/Qwen3-32B` 可修改为自定义路径，确保该路径有足够的磁盘空间（约 62GB）。
- 下载时间可能因网络环境而异，建议在稳定的网络环境下操作。

### 下载镜像

执行以下 Shell 命令，拉取昇思 MindSpore Qwen3 推理容器镜像：

```bash
docker pull swr.cn-central-221.ovaijisuan.com/mindformers/qwen3_mindspore2.6.0-infer:20250428
```

### 启动容器

执行以下命令创建并启动容器：

```bash
docker run -it --privileged  --name=Qwen3 --net=host \
   --shm-size 500g \
   --device=/dev/davinci0 \
   --device=/dev/davinci1 \
   --device=/dev/davinci2 \
   --device=/dev/davinci3 \
   --device=/dev/davinci4 \
   --device=/dev/davinci5 \
   --device=/dev/davinci6 \
   --device=/dev/davinci7 \
   --device=/dev/davinci_manager \
   --device=/dev/hisi_hdc \
   --device /dev/devmm_svm \
   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
   -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
   -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
   -v /usr/local/sbin:/usr/local/sbin \
   -v /etc/hccn.conf:/etc/hccn.conf \
   -v /home:/home \
   swr.cn-central-221.ovaijisuan.com/mindformers/qwen3_mindspore2.6.0-infer:20250428
   /bin/bash
```

#### 注意事项：

- 如果部署在多机上，每台机器中容器的hostname不能重复。如果有部分宿主机的hostname是一致的，需要在起容器的时候修改容器的hostname。
- 后续所有操作均在容器内操作。

## 服务化部署

### 1. 修改模型配置文件

在 `predict_qwen3_32b.yaml` 中对以下配置进行修改（若为默认路径则无需修改）：

```yaml
load_checkpoint: '/home/work/Qwen3-32B/weights'         # 配置为实际的模型权重绝对路径
auto_trans_ckpt: True                                          # 打开权重自动切分，自动将权重转换为分布式任务所需的形式
load_ckpt_format: 'safetensors'
processor:
  tokenizer:
    vocab_file: "/home/work/Qwen3-32B/vocab.json"       # 配置为词表文件的绝对路径
    merges_file: "/home/work/Qwen3-32B/merges.txt"      # 配置为词表文件的绝对路径
```

### 2. 添加环境变量

在服务器中添加如下环境变量：

```bash
export MINDFORMERS_MODEL_CONFIG=/home/work/Qwen3-32B/predict_qwen3_32b.yaml
export ASCEND_CUSTOM_PATH=$ASCEND_HOME_PATH/../
export vLLM_MODEL_BACKEND=MindFormers
export vLLM_MODEL_MEMORY_USE_GB=50
export ASCEND_TOTAL_MEMORY_GB=64
export MS_ENABLE_LCCL=off
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=enp189s0f0
export GLOO_SOCKET_IFNAME=enp189s0f0
export TP_SOCKET_IFNAME=enp189s0f0
export HCCL_CONNECT_TIMEOUT=3600
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### 3. 拉起服务

执行以下命令拉起服务：

```bash
python3 -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server --model "Qwen3-32B" --trust_remote_code --tensor_parallel_size=2 --enable-prefix-caching --enable-chunked-prefill --max-num-seqs=256 --block-size=32 --max_model_len=70000 --max-num-batched-tokens=2048 --distributed-executor-backend=ray
```

### 4. 执行推理请求测试

执行以下命令发送流式推理请求进行测试：

```bash
curl -w "\ntime_total=%{time_total}\n" -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{"inputs": "请介绍一个北京的景点", "parameters": {"do_sample": false, "max_new_tokens": 128}, "stream": false}' http://127.0.0.1:1025/generate_stream &
```

## 声明

本文档提供的模型代码、权重文件和部署镜像，当前仅限于基于昇思MindSpore AI框架体验Qwen3-32B的部署效果，不支持生产环境部署。相关使用问题请反馈至[Issue](https://gitee.com/mindspore/mindformers/issues/new)。
