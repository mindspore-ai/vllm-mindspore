#!/bin/bash
export PYTHONPATH=/yourpath/mindformers/:${PYTHONPATH}
export MINDFORMERS_MODEL_CONFIG=/yourpath/mindformers/research/qwen2_5/predict_qwen2_5_7b_instruct.yaml

MODEL_PATH="/yourpath/Qwen2.5-7B-Instruct"

# Activate Collocation
export ENABLE_COLLOCATION=True

export FORCE_EAGER="true"
export GLOG_v=2
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V1=0 
export MS_ENABLE_LCCL=on
export MS_DEV_RUNTIME_CONF='comm_init_lccl_only:true'
export LCAL_COMM_ID=127.0.0.1:15072
export MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST='MatMulAllReduce'
export VLLM_MS_MODEL_BACKEND="MindFormers"




python3 -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server --model="${MODEL_PATH}" --tensor_parallel_size=2 --max_model_len=2048 --port=24500 --gpu_memory_utilization=0.45

