<h1 align="center">
Collocation
</h1>

# BenchMarks
Running Benchmarks examples

## Init shard memory
The first usage requires manually initializing the shard memory to enable the Collocator:

```
python vllm_mindspore/collocation/collocation_memory_init.py
```

### Apply Qwen2.5 Patch to MindFormers
This script applies the Qwen2.5 patch.
You need to manually apply the patch file from the collocation_mindformers_patch folder to the MindFormers directory:

```
cd mindformers
git apply collocation_mindformers_patch/0001-Qwen2.5-infer-used-INTERNAL-CUSTOM-PagedAttention.patch
```

### Adaptation to do:
#### bench_collocation.sh:

```
MODEL="/yourpath/Qwen2.5-7B-Instruct"
```

##### run_vllm_xxx_server.sh:

```
export PYTHONPATH=/yourpath/mindformers/:${PYTHONPATH}
export MINDFORMERS_MODEL_CONFIG=/yourpath/mindformers/research/qwen2_5/predict_qwen2_5_7b_instruct.yaml

MODEL_PATH="/yourpath/Qwen2.5-7B-Instruct"
```

### Collocation benchmark
To start online vllm-ms server:
First server:

```
bash run_vllm_colloc_0_server.sh
```

_Recommand to launch 2nd server after weight loading of 1st server_
Second server:

```
bash run_vllm_colloc_1_server.sh
```

Launch benchmark:

```
bash bench_collocation.sh Parameter1 Parameter2 Parameter3
```

Parameter1: Request rate for 1st Server
Parameter2: Request rate for 2nd Server
Parameter3: Total requests number
