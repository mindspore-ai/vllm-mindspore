<h1 align="center">
Collocation Showcase 脚本
</h1>

# BenchMarks
Running Benchmarks examples

## 初始化shard memory
初次使用时需要手动初始化shard memory来使能Collocator

```
python vllm_mindspore/collocation/collocation_memory_init.py
```

### 应用Qwen2.5 patch到mindformers目录下
本script应用于Qwen2.5 需要手动应用collocation_mindformers_patch文件夹中的patch文件到Mindformer目录下

```
cd mindformers
git apply collocation_mindformers_patch/0001-Qwen2.5-infer-used-INTERNAL-CUSTOM-PagedAttention.patch
```

### 运行前需要对权重位置，mindformer路径进行适配:
#### bench_collocation.sh:
地址指向Qwen2_5 hugging face权重文件夹

```
MODEL="/yourpath/Qwen2.5-7B-Instruct"
```

##### run_vllm_xxx_server.sh:
地址指向Mindformer以及Qwen2_5 yaml路径

```
export PYTHONPATH=/yourpath/mindformers/:${PYTHONPATH}
export MINDFORMERS_MODEL_CONFIG=/yourpath/mindformers/research/qwen2_5/predict_qwen2_5_7b_instruct.yaml

MODEL_PATH="/yourpath/Qwen2.5-7B-Instruct"
```

### Collocation benchmark
首先开启在线Server:
第一个Server:

```
bash run_vllm_colloc_0_server.sh
```

_推荐第一个Server结束加载权重后再开启第二个Server_
第二个Server:

```
bash run_vllm_colloc_1_server.sh
```

开启show case脚本发送请求:

```
bash bench_collocation.sh 参数1 参数2 参数3
```

参数1: 第一个Server的Request rate，默认1.0
参数2: 第二个Server的Request rate，默认1.0
参数3: 发送的Request总数，默认100
