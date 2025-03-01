# vllm_mindspore

## 项目介绍

功能介绍

API

---

## 前置依赖

### 运行环境

OS：linux aarch64

python：python3.9-3.11

device：Ascend A2/A3卡

软件：

1. `CANN>=8.0.0`
2. `mindspore>=2.5.0`

### 环境验证

`python -c "import mindspore;mindspore.set_context(device_target='Ascend');mindspore.run_check()"`

> **若报错显示CANN不匹配，则需重新安装CANN包**
>
> 则可通过如下方法找到对应的CANN:
>
> 1. 找到包安装路径下的 `.commit_id` (如: `/your_path/site-packages/mindspore/.commit_id`）, 可获取其对应的代码 commit记录，如:
>
>   ```
> __commit_id__ = '[sha1]:94cf8828,[branch]:(HEAD,origin/master,origin/HEAD,master)'
>   ```
>
> 2. 通过该 commit id 查找源码中对应的 `./.jenkins/task/config/cann_version.txt` 文件即可知配套 CANN 的归档日期，重新安装对应CANN。
>
> 
>
> **mindspore，CANN包推荐**
>
> 依赖的 CANN 和 MindSpore 仍处于开发态未正式发布，可通过 
> https://repo.mindspore.cn/mindspore/mindspore/version  获取每日构建版本，并安装对应的 CANN 配套环境。
>
> **推荐获取: Milan_C20_20241231 的CANN，和 20250125 的每日 mindspore 包。**
> mindspore包地址：
> https://repo.mindspore.cn/mindspore/mindspore/version/202501/20250125/master_20250125160017_3f1def978242de1dda3ef0544e282b6ef369d165_newest/unified/aarch64/
>
> CANN包地址：
> https://mindspore-repo.csi.rnd.huawei.com/productrepo

---

## 安装

### 源码安装

```shell
# 1. 安装vllm（已安装有对应版本vllm时可跳过）
git clone https://github.com/vllm-project/vllm.git -b v0.6.6.post1 --depth 1 vllm-v0.6.6.post1
cd vllm-v0.6.6.post1
export VLLM_TARGET_DEVICE=empty
pip install .

# 2. 安装vllm_mindspore
git clone https://gitee.com/mindspore/vllm_mindspore.git
cd vllm_mindspore
pip3 install .

# 3. 卸载torch相关包
pip3 uninstall torch torch-npu torchvision # 卸载 torch 相关包，msadaptor的限制
```

### 通过镜像使用

````
bash build_image.sh $DEVICE_TYPE $VERSION
# bash build_image.sh 800I 2.0.RC1.B020
````

> DEVICE_TYPE可以取值`300I`、`800I`、`A3`
>
> VERSION取值为MindIE版本号，如2.0.RC1.B020

---

## 部署

1. 离线批量推理

   ```python
   import vllm_mindspore # Add this line on the top of script.
   from vllm import LLM, SamplingParams
   
   # Sample prompts.
   prompts = [
       "I am",
       "Today is",
       "Llama is"
   ]
   
   # Create a sampling params object.
   sampling_params = SamplingParams(temperature=0.0, top_p=0.95)
   
   # Create an LLM.
   llm = LLM(model="meta-llama/Llama-2-7b-hf")
   # Generate texts from the prompts. The output is a list of RequestOutput objects
   # that contain the prompt, generated text, and other information.
   outputs = llm.generate(prompts, sampling_params)
   # Print the outputs.
   for output in outputs:
       prompt = output.prompt
       generated_text = output.outputs[0].text
       print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
   ```

   > **关于权重设置**
   >
   > 1. 在线权重需要设置HF_TOKEN
   >
   >    `export HF_TOKEN=Your_token`
   >
   > 2. 本地权重设置
   >
   >    如果已经有下载好的模型配置、权重等，将 `meta-llama/Llama-2-7b-hf` 替换为本地的路径即可。
   >
   >
   > 
   >
   > **https请求失败时额外设置**
   >
   > 由于一些限制，在线下载在特定的服务器上需要通过安装较低版本的 requests 包 `requests-2.27.1`，且需要在脚本最上方添加如下代码:
   >
   > ```python
   > import urllib3
   > import os
   > # disable SSL certificate verification
   > os.environ['CURL_CA_BUNDLE'] = ''
   > # disable_warning
   > urllib3.disable_warnings()
   > ```

2. 服务化（兼容openai）

   **拉起服务** 

   `python3 -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server --model "meta-llama/Llama-2-7b-hf"`

   **发起请求** 

   ```shell
   curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "meta-llama/Llama-2-7b-hf",
       "prompt": "Llama is",
       "max_tokens": 120,
       "temperature": 0
     }'
   ```
   
   



