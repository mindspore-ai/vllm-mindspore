#!/bin/bash

date
npu-smi info

# you should set path when run with crond
#export PATH=/path/to/your/python:$PATH

# you should set aisbench source code path when run cveal and gsm8k
# the path should be {aisbench_source}/benchmark
export AIS_BENCH_PATH=/path/to/your/aisbench

# set this when libdrvdsmi_host.so not found
#export LD_LIBRARY_PATH=/path/to/your/ascend/driver/lib64/driver:$LD_LIBRARY_PATH

# update mindspore/mindformers/msadapter/golden-stick/vllm-mindspore to daily whl
#source /your/proxy.sh
#bash update_packages.sh
unset http_proxy https_proxy

ONE_CARD=0
TWO_CARD=0,1
FOUR_CARD=0,1,2,3
EIGHT_CARD=0,1,2,3,4,5,6,7
QWEN2_7B=/path/to/your/ckpt
QWEN2_32B=/path/to/your/ckpt
QWEN3_32B=/path/to/your/ckpt
TELECHAT2_7B=/path/to/your/ckpt
TELECHAT2_35B=/path/to/your/ckpt
DEEPSEEK_W8A8=/path/to/your/ckpt

# qwen2.5-7b-mindformers
export VLLM_MS_MODEL_BACKEND=MindFormers && export ASCEND_RT_VISIBLE_DEVICES=$ONE_CARD && python benchmark_to_dashboard.py --model $QWEN2_7B --serve-args="--trust-remote-code" --bench-args="--num-prompts=1 --random-input-len=256 --random-output-len=256 --dataset-name=random --trust-remote-code;--num-prompts=192 --random-input-len=256 --random-output-len=256 --dataset-name=random --trust-remote-code" --display-name "Qwen2.5-7B-Instruct-1-256-256;Qwen2.5-7B-Instruct-192-256-256"

# qwen2.5-7b-native
export VLLM_MS_MODEL_BACKEND=Native && export ASCEND_RT_VISIBLE_DEVICES=$ONE_CARD && python benchmark_to_dashboard.py --model $QWEN2_7B --serve-args="--trust-remote-code" --bench-args="--num-prompts=1 --random-input-len=256 --random-output-len=256 --dataset-name=random --trust-remote-code;--num-prompts=192 --random-input-len=256 --random-output-len=256 --dataset-name=random --trust-remote-code" --display-name "Qwen2.5-7B-Instruct-native-1-256-256;Qwen2.5-7B-Instruct-native-192-256-256"

# telechat2_7b
export VLLM_MS_MODEL_BACKEND=MindFormers && export ASCEND_RT_VISIBLE_DEVICES=$ONE_CARD && python benchmark_to_dashboard.py --model=$TELECHAT2_7B --serve-args="--trust-remote-code" --bench-args="--num-prompts=1 --random-input-len=256 --random-output-len=256 --dataset-name=random --trust-remote-code;--num-prompts=192 --random-input-len=256 --random-output-len=256 --dataset-name=random --trust-remote-code" --display-name "telechat2_7b-1-256-256;telechat2_7b-192-256-256"

# telechat2_35b
export VLLM_MS_MODEL_BACKEND=MindFormers && export ASCEND_RT_VISIBLE_DEVICES=$TWO_CARD && python benchmark_to_dashboard.py --model=$TELECHAT2_35B --serve-args="--trust_remote_code --tensor_parallel_size=2 --max-num-seqs=256 --max_model_len=10000 --block-size=32 --gpu-memory-utilization=0.9" --bench-args="--num-prompts=1 --random-input-len=256 --random-output-len=256 --dataset-name=random --trust-remote-code;--num-prompts=192 --random-input-len=256 --random-output-len=256 --dataset-name=random --trust-remote-code" --display-name "telechat2-35b-1-256-256;telechat2-35b-192-256-256"

# qwen2.5-32b-mindformers
export VLLM_MS_MODEL_BACKEND=MindFormers && export ASCEND_RT_VISIBLE_DEVICES=$FOUR_CARD && python benchmark_to_dashboard.py --model $QWEN2_32B --serve-args="--trust-remote-code --tensor-parallel-size 4 --block-size 128 --max-model-len 32768 --max-num-seqs 200 --gpu-memory-utilization 0.9 --cpu-offload-gb 0 --no-enable-prefix-caching --disable-log-requests --disable-log-stats" --bench-args="--num-prompts=1 --random-input-len=1024 --random-output-len=256 --dataset-name=random --trust-remote-code;--num-prompts=1 --random-input-len=8192 --random-output-len=256 --dataset-name=random --trust-remote-code;--num-prompts=8 --random-input-len=8192 --random-output-len=256 --dataset-name=random --trust-remote-code;--num-prompts=16 --random-input-len=8192 --random-output-len=256 --dataset-name=random --trust-remote-code;--num-prompts=32 --random-input-len=1024 --random-output-len=256 --dataset-name=random --trust-remote-code" --display-name "Qwen2.5-32B-Instruct-1-1024-256;Qwen2.5-32B-Instruct-1-8192-256;Qwen2.5-32B-Instruct-8-8192-256;Qwen2.5-32B-Instruct-16-8192-256;Qwen2.5-32B-Instruct-32-1024-256"

# qwen3-32b-native
export VLLM_MS_MODEL_BACKEND=Native && export ASCEND_RT_VISIBLE_DEVICES=$FOUR_CARD && python benchmark_to_dashboard.py --model $QWEN3_32B --serve-args="--max-num-seqs 256 --max-num-batched-tokens 4096 --max_model_len 65536 --block-size 128 --gpu-memory-utilization 0.9 --tensor-parallel-size 4 --trust_remote_code --rope-scaling '{\"rope_type\":\"yarn\",\"factor\":2.0,\"original_max_position_embeddings\":32768}'" --bench-args="--num-prompts=1 --random-input-len=256 --random-output-len=256 --dataset-name=random --trust-remote-code;--num-prompts=192 --random-input-len=256 --random-output-len=256 --dataset-name=random --trust-remote-code" --display-name "Qwen3-32B-native-1-256-256;Qwen3-32B-native-192-256-256"

# deepseek-w8a8-10layer-4card
export VLLM_MS_MODEL_BACKEND=MindFormers && export ASCEND_RT_VISIBLE_DEVICES=$FOUR_CARD && python benchmark_to_dashboard.py --model=$DEEPSEEK_W8A8 --serve-args="-q='ascend' --trust_remote_code --tensor_parallel_size=4 --max_model_len=32768 --max-num-seqs=256 --block-size=128 --gpu-memory-utilization=0.7" --bench-args="--num-prompts=1 --random-input-len=256 --random-output-len=256 --dataset-name=random --trust-remote-code;--num-prompts=192 --random-input-len=256 --random-output-len=256 --dataset-name=random --trust-remote-code" --display-name "deepseek-w8a8-10layer-4card-1-256-256;deepseek-w8a8-10layer-4card-192-256-256"

# deepseek-w8a8-10layer-8card-dp4_tp2_ep4
export VLLM_MS_MODEL_BACKEND=MindFormers && export ASCEND_RT_VISIBLE_DEVICES=$EIGHT_CARD && python benchmark_to_dashboard.py --model=$DEEPSEEK_W8A8 --serve-args="--trust_remote_code --max-num-seqs=8 --max_model_len=4096 --max-num-batched-tokens=8 --block-size=128 --gpu-memory-utilization=0.7 --quantization ascend --tensor-parallel-size 2 --data-parallel-size 4 --data-parallel-size-local 4 --data-parallel-start-rank 0 --data-parallel-address 127.0.0.1 --data-parallel-rpc-port 50051 --enable-expert-parallel --additional-config '{\"expert_parallel\": 4}'" --bench-args="--num-prompts=1 --random-input-len=256 --random-output-len=256 --dataset-name=random --trust-remote-code;--num-prompts=192 --random-input-len=256 --random-output-len=256 --dataset-name=random --trust-remote-code" --display-name "deepseek-w8a8-10layer-dp4_tp2_ep4-1-256-256;deepseek-w8a8-10layer-dp4_tp2_ep4-192-256-256"

