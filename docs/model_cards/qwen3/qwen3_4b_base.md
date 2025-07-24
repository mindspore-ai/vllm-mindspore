# Qwen3-4B-Base vLLM+MindSpore 推理指南

<p align="left">
</p>

## 目录

- [模型介绍](#模型介绍)
- [快速开始](#快速开始)
- [声明](#声明)

## 模型介绍

Qwen 大模型系列的新一代版本 —— Qwen3，在自然语言处理和多模态能力方面实现了全新突破。继承前代模型的成功经验，Qwen3 系列采用了更大规模的数据集、改进的模型架构以及更优的微调技术，使其能够应对更加复杂的推理、语言理解和生成任务。这一代模型还扩展了支持的最大 token 数量，能够生成更长、更连贯的回答，并更好地处理复杂的对话流程。

## 快速开始

当前支持的硬件为Atlas 800T A2服务器

### 下载模型权重
执行以下命令为自定义下载路径`/home/qwen3`添加白名单：

```shell
export HUB_WHITE_LIST_PATHS=/home/qwen3
```

> `/home/qwen3` 可修改为自定义路径，需同步修改后续操作中的下载路径，并确保该路径有足够的磁盘空间。

执行以下 Python 脚本从魔乐社区下载 MindSpore版本的Qwen3 权重及文件至指定路径`/home/qwen3`

```python
from openmind_hub import snapshot_download

snapshot_download(
    repo_id="MindSpore-Lab/Qwen3-4B-Base",
    local_dir="/home/qwen3",
    local_dir_use_symlinks=False
)
```

### 下载镜像

执行以下 Shell 命令，拉取MindSpore Qwen3 推理容器镜像：

```sh
docker pull swr.cn-central-221.ovaijisuan.com/mindsporelab/mindspore2.6.0-cann7.6.0.1-python3.11-openeuler22.03:v1
```

### 创建并进入容器

执行以下命令创建容器,name设置为qwen3：

```sh
docker run -itd --privileged  --name=qwen3 --net=host \
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
   -v /home/qwen3:/home/qwen3 \
   swr.cn-central-221.ovaijisuan.com/mindsporelab/mindspore2.6.0-cann7.6.0.1-python3.11-openeuler22.03:v1 \
   bash
```

进入容器，后续所有操作均在容器内操作

```
docker exec -it qwen3 bash
```

### 执行推理服务

```shell
python generate_vllm.py --model_path='/home/qwen3'
```

```python
# generate_vllm.py
import vllm_mindspore # Add this line on the top of script.
import mindspore

from vllm import LLM, SamplingParams


def main(args):
    # Sample prompts.
    prompts = [
        "MindSpore is",
        "Qwen3 is",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=args.max_tokens)

    # Create an LLM.
    llm = LLM(model=args.model_path)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="vllm-mindspore qwen3 demo")
    parser.add_argument("--model_path", type=str, default="Qwen3-4B-Base")
    args, _ = parser.parse_known_args()

    main(args)
```

### 性能如下：

| model_name        |precision | tokens/s |
|-------------------|   ---    |----------|
| Qwen3-4B-Base     |  bf16    | 26.48    |
