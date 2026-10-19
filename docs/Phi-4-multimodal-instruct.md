# 环境依赖安装

> 参考链接：[安装指南 | vLLM-MindSpore Plugin 0.4.0 文档 | 昇思MindSpore社区](https://www.mindspore.cn/vllm_mindspore/docs/zh-CN/r0.4.0/getting_started/installation/installation.html)

- 依赖版本列表

| 软件                   | 配套版本与下载链接                                           |
| ---------------------- | ------------------------------------------------------------ |
| CANN                   | [8.3.RC1](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/index/index.html) |
| MindSpore              | [ 2.7.1](https://www.mindspore.cn/versions#2.7.1)            |
| MSAdapter              | [ 0.5.0](https://repo.mindspore.cn/mindspore/msadapter/version/202510/20251011/r0.3.0_20251011095813_951a8218d4c29785e48f304e720212b57056573e_newest/) |
| MindSpore Transformers | [ 1.7.0](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.7.0) |
| vLLM                   | [0.11.0](https://repo.mindspore.cn/mirrors/vllm/version/202511/20251113/v0.11.0/any/) |
| Transformer            | [4.57.1](https://files.pythonhosted.org/packages/71/d3/c16c3b3cf7655a67db1144da94b021c200ac1303f82428f2beef6c2e72bb/transformers-4.57.1-py3-none-any.whl) |
| mindone                | [master](https://github.com/mindspore-lab/mindone)           |


# 数据集说明

## 1. 数据集下载

Phi-4 多模态模型使用的官方权重与相关配置文件可从 Hugging Face 下载：

https://huggingface.co/microsoft/Phi-4-multimodal-instruct

下载完成后，请将模型文件放置在本地模型目录中，用于后续模型加载与推理。

---

## 2. 数据处理说明

由于当前适配版本与 Hugging Face 原始配置存在部分兼容性问题，需要对模型目录中的配置文件进行简单修改。

### （1）修改 `preprocessor_config.json`

在 `preprocessor_config.json` 文件中 **删除以下 `auto_map` 字段内容**：

```json
"auto_map": {
  "AutoProcessor": "processing_phi4mm.Phi4MMProcessor",
  "AutoImageProcessor": "processing_phi4mm.Phi4MMImageProcessor",
  "AutoFeatureExtractor": "processing_phi4mm.Phi4MMAudioFeatureExtractor"
}
```


### （2）修改 `processor_config.json`

在 `processor_config.json` 文件中 **删除以下 `auto_map` 字段内容**：
```
"auto_map": {
  "AutoProcessor": "processing_phi4mm.Phi4MMProcessor"
}
```


# 推理示例说明
> 模型支持文本、图像和音频三种模态的推理。其中，图像与音频模态的推理需要在加载对应的 LoRA 权重后才能启用相关能力。
## 离线推理
纯文本：
```
import vllm_mindspore

TEXT_PROMPT = (
    "<|user|>"
    "what is the answer for 1+1? Explain it."
    "<|end|>"
    "<|assistant|>"
)

model_path = "/root/model/update/phi4"

def generate_llm_engine(enforce_eager=True, tensor_parallel_size=1):
    from vllm import LLM
    # Create an LLM.
    llm = LLM(model=model_path,
              gpu_memory_utilization=0.7,
              tensor_parallel_size=tensor_parallel_size,
              enforce_eager=enforce_eager,
              max_model_len=4096,
              max_num_seqs=32,
              max_num_batched_tokens=4096,
              trust_remote_code=True, 
            #   limit_mm_per_prompt={"image": 0, "audio": 0}
              )

    return llm

def test_phi4_text_only():
    """
    Test the text-only inference capabilities of the Phi-4 multimodal model
    """
    import vllm_mindspore
    from vllm import LLM, SamplingParams

    inputs = [{
        "prompt": TEXT_PROMPT,
        "multi_modal_data": {}  
    }]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024, top_k=1)

    llm = generate_llm_engine(enforce_eager=True, tensor_parallel_size=1)

    outputs = llm.generate(inputs, sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    test_phi4_text_only()
```

文本+图像：
```
import vllm_mindspore

import os

from PIL import Image

from vllm.lora.request import LoRARequest

VISION_PROMPT = (
    "<|user|>"
    "<|image_1|>"
    "What is shown in this image? Describe in detail."
    "<|end|>"
    "<|assistant|>"
)

image_path = "/root/gitee/vllm-mindspore/atemp/test_phi4/examples/image.png"

model_path = "/root/model/update/phi4"
speech_lora_path = "/root/model/update/phi4/speech-lora"
vision_lora_path = "/root/model/update/phi4/vision-lora"

def pil_image(image_path=image_path) -> Image.Image:
    return Image.open(image_path)

def generate_llm_engine(enforce_eager=True, tensor_parallel_size=1):
    from vllm import LLM
    # Create an LLM.
    llm = LLM(model=model_path,
              gpu_memory_utilization=0.7,
              tensor_parallel_size=tensor_parallel_size,
              enforce_eager=enforce_eager,
              max_model_len=4096,
              max_num_seqs=32,
              max_num_batched_tokens=4096,
              trust_remote_code=True,  
              enable_lora=True,
              max_lora_rank=256,
              lora_extra_vocab_size=256,
              max_loras=2,
            #   limit_mm_per_prompt={"image": 0, "audio": 0}
              )

    return llm



def test_phi4_vision():
    """
    Test the image + text multimodal inference capabilities of the Phi-4 model
    """
    from vllm import LLM, SamplingParams

    image = pil_image()
    inputs = [{
        "prompt": VISION_PROMPT,
        "multi_modal_data": {"image": [image]}  
    }]
    speech_lora = LoRARequest("speech_adapter", 1, speech_lora_path)
    vision_lora = LoRARequest("vision_adapter", 2, vision_lora_path)

   
    sampling_params = SamplingParams(temperature=0.7, max_tokens=512,top_k=1)
    
    
    llm = generate_llm_engine(enforce_eager=True, tensor_parallel_size=1)

   
    outputs = llm.generate(inputs, sampling_params,lora_request=vision_lora)

   
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    test_phi4_vision()

```

文本+音频：
```
import vllm_mindspore

import soundfile

from vllm.lora.request import LoRARequest

AUDIO_PROMPT = (
    "<|user|>"
    "<|audio_1|>"
    "Based on the attached audio, generate a comprehensive text transcription of the spoken content."
    "<|end|>"
    "<|assistant|>"
)

audio_path = "/root/gitee/vllm-mindspore/atemp/test_phi4/examples/what_is_shown_in_this_image.wav"

model_path = "/root/model/update/phi4"
speech_lora_path = "/root/model/update/phi4/speech-lora"
vision_lora_path = "/root/model/update/phi4/vision-lora"

def load_audio(path: str):
    """
    Load audio file as a numpy array
    """
    audio = soundfile.read(path) 
    return audio

def generate_llm_engine(enforce_eager=True, tensor_parallel_size=1):
    from vllm import LLM
    # Create an LLM.
    llm = LLM(model=model_path,
              gpu_memory_utilization=0.7,
              tensor_parallel_size=tensor_parallel_size,
              enforce_eager=enforce_eager,
              max_model_len=4096,
              max_num_seqs=32,
              max_num_batched_tokens=4096,
              trust_remote_code=True,  
              enable_lora=True,
              max_lora_rank=320,
              lora_extra_vocab_size=256,
              max_loras=2,
            #   limit_mm_per_prompt={"image": 0, "audio": 0}
              )

    return llm




def test_phi4_audio_text():
    """
    Test the audio + text multimodal inference capabilities of the Phi-4 model
    """
    from vllm import LLM, SamplingParams

    
    audio = load_audio(audio_path)
    inputs = [{
        "prompt": AUDIO_PROMPT,
        "multi_modal_data": {"audio": [audio]}  
    }]

    speech_lora = LoRARequest("speech_adapter", 1, speech_lora_path)
    vision_lora = LoRARequest("vision_adapter", 2, vision_lora_path)


    
    sampling_params = SamplingParams(temperature=0.4, max_tokens=1024, top_k=1)

    
    llm = generate_llm_engine(enforce_eager=True, tensor_parallel_size=1)

    
    outputs = llm.generate(inputs, sampling_params,lora_request=speech_lora)

    
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
   
    test_phi4_audio_text()

```
## 在线推理
启动服务：
不加载lora
```cmd
vllm-mindspore serve /root/model/update/phi4 --gpu-memory-utilization 0.7 --tensor-parallel-size 1 --trust-remote-code
```

启动服务：加载lora
```cmd
vllm-mindspore serve /root/model/update/phi4 --gpu-memory-utilization 0.7 --tensor-parallel-size 1 --trust-remote-code --max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 4096 --enable-lora --max-lora-rank 320 --lora-extra-vocab-size 256 --max-loras 2 --enable-log-requests --lora-modules speech_adapter=/root/model/update/phi4/speech-lora
```

请求推理：纯文本
```python
import requests
import json

def test_phi4_online_text_only_chat():
    """
    测试 Phi-4 多模态模型的纯文本 Chat 模式在线推理能力
    """
    
    url = "http://localhost:8000/v1/chat/completions"
    
    messages = [
        {
            "role": "user",
            "content": "what is the answer for 1+1? Explain it."
        }
    ]

    payload = {
        "model": "/root/model/update/phi4",  
        "messages": messages,                 
        "max_tokens": 1024,
        "temperature": 0.0,
        "top_k": 1,
        "stream": False 
    }

    headers = {
        "Content-Type": "application/json"
    }

    print(f"🚀 正在发送 Chat 模式请求到 Phi-4 服务器...")
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]
            print("-" * 30)
            print(f"Generated text: {generated_text}")
            print("-" * 30)
        else:
            print(f"❌ 请求失败，状态码: {response.status_code}")
            print(f"错误详情: {response.text}")
            
    except Exception as e:
        print(f"❌ 发生异常: {e}")

if __name__ == "__main__":
    test_phi4_online_text_only_chat()
```

请求推理：文本+图像
```python
import requests
import json
import base64
from PIL import Image
import io

# Chat 模式的消息格式
def create_vision_chat_messages(image_base64: str, user_prompt: str):
    """
    构造包含图片的 chat 消息列表（OpenAI Vision 格式）
    """
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        }
    ]

def encode_image_to_base64(image_path):
    """将图片文件转换为 Base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_phi4_chat_vision_lora():
    """
    测试 Phi-4 模型 Chat 模式 + 图片 + Vision LoRA 的在线推理能力
    """
    # 🔹 Chat 模式端点（注意是 /chat/completions）
    url = "http://localhost:8000/v1/chat/completions"
    image_path = "/root/gitee/vllm-mindspore/atemp/test_phi4/examples/image.png"
    
    # 1. 准备图片 Base64
    base64_image = encode_image_to_base64(image_path)

    # 2. 构造 Chat 消息
    messages = create_vision_chat_messages(
        image_base64=base64_image,
        user_prompt="What is shown in this image? Describe in detail."
    )

    # 3. 构造请求 Payload（OpenAI Chat 兼容格式）
    payload = {
        "model": "vision_adapter",  # 或你的 LoRA 适配器名称
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_k": 1,
        "stream": False
        # 可选参数:
        # "top_p": 0.9,
        # "frequency_penalty": 0.0,
        # "presence_penalty": 0.0,
    }

    headers = {"Content-Type": "application/json"}

    print(f"🚀 正在发送 Chat 模式图片请求到 Phi-4 服务器...")

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            print("-" * 30)
            # 🔹 Chat 模式响应解析路径不同
            generated_text = result["choices"][0]["message"]["content"]
            print(f"✅ Generated response:\n{generated_text}")
            print("-" * 30)
            
            # 可选：打印完整响应用于调试
            # print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"❌ 请求失败，状态码: {response.status_code}")
            print(f"详情: {response.text}")
            
    except Exception as e:
        print(f"❌ 发生异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_phi4_chat_vision_lora()
```

请求推理：文本+语音
```python
import requests
import json
import base64
import soundfile

def test_phi4_online_audio_chat():
    """
    测试 Phi-4 多模态模型的音频+文本 Chat 模式在线推理
    """
    # 👇 1. 端点改为 chat/completions
    url = "http://localhost:8000/v1/chat/completions"

    audio_path = "/root/gitee/vllm-mindspore/atemp/test_phi4/examples/what_is_shown_in_this_image.wav"

    # 读取音频并 base64 编码（推荐方式，兼容性更好）
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    # 👇 2. 使用 messages 格式构建请求
    messages = [
        {
            "role": "user",
            "content": [
                # 👇 音频部分：使用 audio_url + data URI 格式
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": f"data:audio/wav;base64,{audio_b64}"
                    }
                },
                # 👇 文本部分
                {
                    "type": "text",
                    "text": "Based on the attached audio, generate a comprehensive text transcription of the spoken content."
                }
            ]
        }
    ]

    payload = {
        "model": "speech_adapter",  # 或你的模型名称
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.4,
        "top_k": 1,
        "stream": False
        # 如需其他参数可继续添加：top_p, repetition_penalty 等
    }

    headers = {
        "Content-Type": "application/json"
    }

    print(f"🚀 正在发送 Chat 模式音频+文本请求到 Phi-4 服务器...")

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            result = response.json()
            # 👇 3. Chat 模式响应结构不同
            generated_text = result["choices"][0]["message"]["content"]
            print("-" * 30)
            print(f"Generated text: {generated_text}")
            print("-" * 30)
        else:
            print(f"❌ 请求失败，状态码: {response.status_code}")
            print(f"错误详情: {response.text}")
    except Exception as e:
        print(f"❌ 发生异常: {e}")

if __name__ == "__main__":
    test_phi4_online_audio_chat()
```