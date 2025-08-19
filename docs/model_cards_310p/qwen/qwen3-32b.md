# Qwen3-32B Atlas 300I Duo vLLM+MindSpore 推理指南
- [Qwen3-32B Atlas 300I Duo vLLM+MindSpore 推理指南](#qwen3-32b-atlas-300i-duo-vllmmindspore-推理指南)
  - [下载模型权重](#下载模型权重)
    - [注意事项：](#注意事项)
  - [服务化部署](#服务化部署)
  - [执行推理请求测试](#执行推理请求测试)
  - [附录](#附录)
    - [相关代码仓](#相关代码仓)

## 下载模型权重

从 [https://huggingface.co/Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) 上下载模型权重

下面以下载到`/home/ckpt/Qwen3-32B` 目录为例。

### 注意事项：

- 将 `/home/ckpt/Qwen3-32B/config.json` 修改中 `"torch_dtype": "float16",`
- `/home/ckpt/Qwen3-32B` 可修改为自定义路径，确保该路径有足够的磁盘空间（约 62GB）。
- 下载时间可能因网络环境而异，建议在稳定的网络环境下操作。

## 服务化部署

使用如下脚本直接部署：

```shell
unset vLLM_MODEL_BACKEND
export MS_ENABLE_LCCL=off
export HCCL_OP_EXPANSION_MODE="AI_CPU"

vllm-mindspore serve "/home/ckpt/Qwen3-32B/" --trust_remote_code --tensor_parallel_size=4 --max-num-seqs=256 --block-size=128 --gpu-memory-utilization=0.5 --max_model_len=4096 --max-num-batched-tokens=4096 --max-model-len=8192 --distributed-executor-backend=ray --swap-space=0 --no-enable-prefix-caching 2>&1 | tee log.txt
```

参数含义可参考 [vLLM参数](https://docs.vllm.ai/en/latest/cli/index.html)

## 执行推理请求测试

执行以下命令发起流式推理请求：  
（model项应该设置为和启动服务时的 model 项一致）

```cpp
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/ckpt/Qwen3-32B",
    "prompt": "Mindspore is",
    "max_tokens": 120,
    "temperature": 0
  }'
```

## 附录
### 相关代码仓
当前代码仓配套如下表所示：
|代码仓|代码仓地址|0.9.1配套分支|
|---|---|---|
|mindspore|[https://gitee.com/mindspore/mindspore.git](https://gitee.com/mindspore/mindspore.git)|br_infer_iter|
|vllm|[https://github.com/vllm\-project/vllm.git](https://github.com/vllm-project/vllm.git)|tag:v0.9.1|
|MSAdapter|[https://git.openi.org.cn/OpenI/MSAdapter.git](https://git.openi.org.cn/OpenI/MSAdapter.git)|msa_r0.2.0|
|vllm mindspore|[https://gitee.com/mindspore/vllm\-mindspore.git](https://gitee.com/mindspore/vllm-mindspore.git)|br_infer_boom|
