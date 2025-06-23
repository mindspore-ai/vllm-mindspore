# 盘古Pro MoE vLLM+MindSpore部署指南

## 目录
- [模型介绍](#模型介绍)
- [快速开始](#快速开始)
- [服务化部署](#服务化部署)
- [离线推理部署](#离线推理部署)
- [声明](#声明)

## 模型介绍
盘古Pro MoE是华为于2025年5月28日发布的大语言模型，其基于创新的分组混合专家模型（Mixture of Grouped Experts, MoGE） 架构，在专家选择阶段对专家进行分组，并约束 token 在每个组内激活等量专家，从而实现专家负载均衡，显著提升模型在昇腾平台的部署效率。盘古Pro MoE总参数量720亿、激活参数量160亿，在中英文和逻辑推理任务的多个权威基准上表现卓越，SuperCLUE 2025 年 5 月排行榜上，盘古 Pro MoE 在千亿参数量以内的模型中并列国内第一。

### 下载链接

|  社区  | 下载地址                                                      |
|:----:|:----------------------------------------------------------|
| 魔乐社区 | https://modelers.cn/models/MindSpore-Lab/Pangu-Pro-MoE |


## 快速开始
盘古Pro MoE推理建议使用1台（8卡）Atlas 800T A2（64G）服务器（基于BF16权重）或使用1台（8卡）Atlas 300I Duo服务器（基于BF16权重）。昇思MindSpore提供了盘古Pro MoE推理可用的Docker容器镜像，供开发者快速体验。

### 下载模型权重

执行以下命令将模型权重的自定义下载路径 `/home/work/PanguProMoE` 添加到白名单：

```bash
export HUB_WHITE_LIST_PATHS=/home/work/PanguProMoE
```

执行以下 Python 脚本，从魔乐社区下载昇思 MindSpore 版本的 盘古Pro MoE模型权重文件至指定路径 `/home/work/PanguProMoE` 。下载的文件包含模型配置、模型权重和分词模型，占用约 150GB 的磁盘空间：

```python
from openmind_hub import snapshot_download

snapshot_download(
    repo_id="MindSpore-Lab/Pangu-Pro-MoE",
    local_dir="/home/work/PanguProMoE",
    local_dir_use_symlinks=False
)
```

下载完成的 `/home/work/PanguProMoE` 文件夹目录结构如下：

```text
PanguProMoE
  ├── config.json                         # 模型配置
  ├── tokenization_pangu_moe.py           # 词表代码
  ├── tokenizer.model                     # 词表模型
  ├── special_tokens_map.json             # 特殊词表映射
  ├── model-xxxxx-of-xxxxx.safetensors    # 模型权重文件
  ├── ...
  └── model.safetensors.index.json        # 模型权重映射文件

```

#### 注意事项

- `/home/work/PanguProMoE` 可修改为自定义路径，需要确保该路径有足够的磁盘空间（约 150GB）。
- 下载时间可能因网络环境而异，建议在稳定的网络环境下操作。
- 使用Atlas 300I Duo推理需要将`config.json`文件中的`torch_dtype`配置项由`bfloat16`改为`float16`。对应的BF16权重将在模型加载时，自动转换为FP16权重。

### 下载镜像

若使用Atlas 800T A2进行推理，则需执行以下 Shell 命令，拉取昇思 MindSpore 盘古Pro MoE Atlas 800T A2推理镜像：

```bash
docker pull swr.cn-central-221.ovaijisuan.com/mindsporelab/pangu_pro_moe_mindspore-infer:800-A2-20250623
```

若使用Atlas 300I Duo进行推理，则需执行以下 Shell 命令，拉取昇思 MindSpore 盘古Pro MoE Atlas 300I Duo推理镜像：

```bash
docker pull swr.cn-central-221.ovaijisuan.com/mindsporelab/pangu_pro_moe_mindspore-infer:300I-Duo-20250623
```

### 启动容器

以Atlas 800T A2推理为例，执行以下命令，创建并启动容器：

```bash
docker run -it --privileged --name=pangu_pro_moe --net=host \
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
   -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
   -v /usr/local/sbin:/usr/local/sbin \
   -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
   -v /etc/hccn.conf:/etc/hccn.conf \
   -v /home:/home \
   swr.cn-central-221.ovaijisuan.com/mindsporelab/pangu_pro_moe_mindspore-infer:800-A2-20250623 \
   /bin/bash
```

> 若使用Atlas 300I Duo进行推理，则将上述启动命令中的镜像地址`swr.cn-central-221.ovaijisuan.com/mindsporelab/pangu_pro_moe_mindspore-infer:800-A2-20250623`替换为`swr.cn-central-221.ovaijisuan.com/mindsporelab/pangu_pro_moe_mindspore-infer:300I-Duo-20250623`。

#### 注意事项

- 后续所有操作均在容器内执行。

## 服务化部署

### 1. 拉起服务

执行以下shell命令启动Atlas 800T A2推理服务：

```bash
vllm-mindspore serve "/home/work/PanguProMoE" --tokenizer "/home/work/PanguProMoE" --trust-remote-code --tensor-parallel-size=8 --gpu-memory-utilization=0.9 --max-num-batched-tokens=2048 --max-num-seqs=512 --block-size=128 --max-model-len=32768
```

若使用Atlas 300I Duo进行推理，则需执行以下shell命令启动推理服务：
```bash
vllm-mindspore serve "/home/work/PanguProMoE" --tokenizer "/home/work/PanguProMoE" --trust-remote-code --tensor-parallel-size=8 --gpu-memory-utilization=0.8 --max-num-batched-tokens=2048 --max-num-seqs=128 --block-size=128 --max-model-len=32768
```

### 2. 执行推理请求测试

执行以下命令，发送推理请求进行测试：

```bash
curl http://localhost:8000/v1/completions -H "Content-type: application/json" -d '{"model": "/home/work/PanguProMoE", "prompts": "[unused9]系统：[unused10][unused9]用户：请简单介绍一个北京的景点[unused10][unused9]助手：", "max_new_tokens": 1024, "temperature": 0.0} &
```

系统将返回以下推理结果：
```txt
 [unused16] 嗯，用户让我简单介绍一个北京的景点。首先，我需要确定用户的需求是什么。可能他们计划去北京旅游，或者在做相关的项目，需要快速了解一个景点。用户提到“简单介绍”，所以不需要太详细的信息，但关键点要涵盖到。\n\n接下来，我得选一个北京最有代表性的景点。故宫肯定是首选，因为它不仅是北京的标志性建筑，也是世界文化遗产，知名度高，信息也容易整理。不过，也有可能用户想了解其他景点，比如长城、颐和园或者天坛。但考虑到故宫的综合性，可能更适合作为例子。\n\n然后，我需要确定介绍的结构。通常包括名称、位置、历史背景、主要看点、特色以及参观提示。这样用户能快速获取关键信息。比如故宫的历史背景，从明朝开始，建筑规模，文物收藏，这些都是重点。\n\n还要注意语言简洁，避免使用复杂术语。比如提到“世界最大宫殿建筑群”这样的数据，既准确又易懂。另外，参观提示部分也很重要，比如开放时间和提前预约，这对游客来说很实用。\n\n可能用户没有说出来的深层需求是希望了解如何有效参观故宫，或者有什么特别值得注意的地方。所以除了基本介绍，加入一些实用信息会更好。比如提到最佳参观时间避开节假日，或者推荐路线，但用户要求简单，所以可能不需要太详细。\n\n另外，要检查有没有错误信息。比如故宫的占地面积、房间数量这些数据是否准确。确认无误后，再组织语言，确保流畅自然。最后，用友好的结尾邀请用户进一步提问，增加互动性。 [unused17] 好的！这里为您介绍北京著名的**故宫博物院**（简称故宫）：\n\n**故宫**是中国明清两代的皇家宫殿，位于北京市中心，旧称“紫禁城”，始建于明永乐四年（1406年），历时14年建成。它是世界现存最大、保存最完整的木质结构古建筑群，1987年被列为世界文化遗产。\n\n**主要看点**：  \n1. **建筑群**：以太和殿、中和殿、保和殿为中心，对称布局，体现古代皇家建筑的恢弘与对称美学。  \n2. **文物藏品**：馆藏近200万件文物，包括陶瓷、书画、宫廷器具等，如《清明上河图》真迹曾在此展出。  \n3. **文化体验**：可参与“故宫讲解”或观看《石渠宝笈》等专题展览，感受历史与艺术的交融。  \n\n**特色**：  \n- 占地面积72万平方米，有9999间房屋（实为约8707间）。  \n- 四季景色各异，春天的海棠、秋天的银杏尤为迷人。  \n\n**参观提示**：  \n- 需提前通过官网或小程序预约购票（旺季约60元/人）。  \n- 建议预留半天至一天时间，跟随导览路线深入了解。  \n\n故宫不仅是中国历史的见证，更是全球游客了解中国传统文化的窗口。如果有具体需求，可以进一步探讨哦！
```


## 离线推理部署

执行以下shell命令，执行离线推理程序`generate_vllm.py`：
```shell
python generate_vllm.py --model_path='/home/work/PanguProMoE'
```

离线推理程序`generate_vllm.py`样例如下：
```python
# generate_vllm.py
import vllm_mindspore # Add this line on the top of script.
from vllm import LLM, SamplingParams

sys_prompt = "[unused9]系统：[unused10][unused9]用户：%s[unused10][unused9]助手："

def main(args):
    # Sample prompts.
    prompts = [
        sys_prompt % "请简单介绍Mindspore",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    # Create an LLM.
    llm = LLM(model=args.model_path,
              gpu_memory_utilization=0.9, # Atlas 300I Duo平台需要修改为0.8
              tensor_parallel_size=8,
              max_num_batched_tokens=2048,
              max_num_seqs=512, # Atlas 300I Duo平台需要修改为128
              max_model_len=32768,
              block_size=128,
              trust_remote_code=True
              )

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
    parser = argparse.ArgumentParser(description="vllm-mindspore pangu_pro_moe demo")
    parser.add_argument("--model_path", type=str, default="/home/work/PanguProMoE")
    parser.add_argument("--max_tokens", type=int, default=1024)
    args, _ = parser.parse_known_args()

    main(args)
```

系统将返回以下推理结果：
```txt
 [unused16] 嗯，用户让我简单介绍一下MindSpore。首先，我需要确定用户对这个框架的了解程度。可能他们听说过TensorFlow或PyTorch，但不太清楚MindSpore有什么不同。我应该先介绍MindSpore的基本信息，比如它是华为开发的，开源的深度学习框架。\n\n接下来，用户可能想知道为什么选择MindSpore而不是其他框架。这时候需要提到它的跨平台特性，支持云端、边缘和移动端，这对现在的AI应用很重要。另外，自动并行和异构计算支持也是华为的优势，特别是结合他们的硬件产品，比如昇腾芯片，这点应该强调。\n\n然后，用户可能关心它的易用性。MindSpore的API设计是否友好，有没有像PyTorch那样的动态图机制，这对开发者来说很重要。还有模型可视化工具，能帮助调试和理解模型结构，这也是一个卖点。\n\n性能方面，自动混合精度和高效推理是关键点，特别是对于企业用户来说，推理速度直接影响应用效果。另外，隐私保护功能，比如联邦学习和差分隐私，可能对处理敏感数据的用户很有用，这也是当前的热点话题。\n\n生态方面，需要提到社区支持和与其他框架的兼容性，比如ONNX和TensorFlow的模型转换，这样用户知道迁移成本不高。最后，适用场景部分，可以举几个例子，比如计算机视觉、自然语言处理，还有华为在边缘计算上的应用，比如手机和物联网设备。\n\n可能用户没有明确说出来的需求是，他们可能在考虑是否从其他框架迁移到MindSpore，或者寻找适合他们项目的框架。所以需要强调MindSpore的优势，比如分布式训练的简便性，跨平台部署，以及华为的硬件支持，这些可能成为他们选择的关键因素。\n\n需要避免使用太多技术术语，保持解释简洁明了。同时，确保覆盖主要特点，让用户有一个全面的了解，但不过于深入细节。最后，总结适用场景，帮助他们判断是否适合他们的需求。 [unused17] MindSpore 是华为公司开发的**开源深度学习框架**，旨在为开发者提供高效、灵活且易于使用的工具，支持人工智能模型的开发、训练和部署。以下是其核心特点和优势：\n\n---\n\n### **核心特性**\n1. **跨平台支持**  \n   - 支持云端、边缘设备（如手机、IoT设备）和云端服务器等多种部署场景，适配华为昇腾（Ascend）芯片、GPU（如英伟达）和CPU。\n   - 提供**异构计算**能力，可自动分配计算资源，优化性能。\n\n2. **自动并行与分布式训练**  \n   - 独创的**自动并行技术**（Auto Parallel），简化大规模分布式训练，提升计算效率。\n   - 支持数据并行、模型并行等多种策略，适合处理复杂模型和大规模数据集。\n\n3. **动态图与静态图融合**  \n   - 结合动态图的灵活性和静态图的优化优势，开发者可通过**Eager模式**（动态图）快速调试，通过**Graph模式**（静态图）提升训练效率。\n\n4. **模型可视化与调试工具**  \n   - 提供模型结构可视化工具（如MindSpore ModelZoo），帮助开发者直观理解模型架构和计算流程。\n\n5. **高效推理与优化**  \n   - 支持自动混合精度（AMP）、算子融合等技术，显著提升推理速度，降低资源消耗。\n\n6. **隐私保护**  \n   - 集成联邦学习、差分隐私等技术，适用于医疗、金融等敏感数据场景。\n\n---\n\n### **优势亮点**\n- **易用性**：API设计简洁，兼容PyTorch、TensorFlow等框架的编程习惯，降低学习成本。\n- **性能优异**：针对华为昇腾芯片深度优化，在昇腾硬件上性能领先。\n- **开源生态**：开放源代码（GitHub托管），拥有活跃社区和丰富的预训练模型库（ModelZoo）。\n- **跨框架兼容**：支持ONNX、TensorFlow模型转换，方便与其他框架协同使用。\n\n---\n\n### **适用场景**\n- **计算机视觉**：图像分类、目标检测、视频分析等。\n- **自然语言处理**：文本生成、机器翻译、情感分析等。\n- **语音处理**：语音识别、语音合成。\n- **边缘AI**：轻量化模型部署到手机、无人机等设备。\n- **科学研究**：如药物研发、气象预测等领域的数值模拟。\n\n---\n\n### **华为生态协同**\nMindSpore 与华为云（ModelArts）、昇腾AI处理器（Ascend）深度集成，提供端到端的AI解决方案，尤其适合需要高性能计算和低延迟部署的场景。\n\n如果需要快速上手，可以参考官方文档和示例代码：[MindSpore官网](https://www.mindspore.cn/)。
```

在昇腾Atlas 800T A2平台上部署盘古Pro MoE模型（基于W8A8量化权重），可在时延100ms的约束条件下，实现平均每卡1020token/s的增量吞吐性能。配套代码和镜像计划于6月底发布，敬请期待。

## 声明
本文档提供的模型代码和镜像，当前仅限用于测试和体验昇思MindSpore盘古Pro MoE模型的推理服务化部署，不建议用于生产环境，正式商用版本计划于Q3发布。如遇使用问题，欢迎反馈至[Issue](https://gitee.com/mindspore/vllm-mindspore/issues/new)。