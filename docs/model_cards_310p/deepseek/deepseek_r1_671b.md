# DeepSeek-R1-671B Atlas 300I Duo vLLM+MindSpore 推理指南
- [DeepSeek-R1-671B Atlas 300I Duo vLLM+MindSpore 推理指南](#deepseek-r1-671b-atlas-300i-duo-vllmmindspore-推理指南)
  - [模型介绍](#模型介绍)
  - [下载模型权重](#下载模型权重)
  - [一键部署](#一键部署)
  - [执行推理请求测试](#执行推理请求测试)
  - [附录](#附录)
    - [相关代码仓](#相关代码仓)
    - [启动脚本](#启动脚本)
    - [配置文件](#配置文件)
  
## 模型介绍

DeepSeek-R1是由深度求索(DeepSeek)公司研发的高性能AI推理模型，基于6710亿参数的大语言模型架构，专注于数学、代码和复杂逻辑推理任务。该模型采用专家混合(MoE)技术，通过强化学习优化推理能力，在AIME数学竞赛和Codeforces编程挑战中表现超越96%人类参与者。

## 下载模型权重

从 [https://modelers.cn/models/MindSpore-Lab/R1-A8W4](https://modelers.cn/models/MindSpore-Lab/R1-A8W4) 上下载模型权重

之后步骤以下载到`/home/ckpt/deepseek_r1_671b` 目录为例。

注意事项：

- `/home/ckpt/deepseek_r1_671b` 可修改为自定义路径，确保该路径有足够的磁盘空间。
- 下载时间可能因网络环境而异，建议在稳定的网络环境下操作。

## 一键部署
可使用一键部署脚本进行部署(内容见附录）:

```
1. 在主节点上运行`source env_main.sh`, 从节点上`source env_sub.sh`设置环境变量。(修改ray启动的ip地址为实际的ip地址）

2. 运行`bash run_vllm.sh`一键启动deepseek服务。(将脚本中启动参数的权重地址修改为实际地址）
```

## 执行推理请求测试

执行以下命令发起流式推理请求：  
（model项应该设置为和启动服务时的 model 项一致）

```shell
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "/home/ckpt/deepseek_r1_671b",
"prompt": "Mindspore is",
"max_tokens": 120,
"temperature": 0
}'
```

## 附录
### 相关代码仓
当前代码仓配套如下表所示

|代码仓|代码仓地址|0.9.1配套分支|
|---|---|---|
|mindspore|[https://gitee.com/mindspore/mindspore.git](https://gitee.com/mindspore/mindspore.git)|br\_infer\_iter|
|mindformers|[https://gitee.com/mindspore/mindformers.git](https://gitee.com/mindspore/mindformers.git)|br_infer_boom|
|vllm|[https://github.com/vllm\-project/vllm.git](https://github.com/vllm-project/vllm.git)|tag:v0.9.1|
|MSAdapter|[https://git.openi.org.cn/OpenI/MSAdapter.git](https://git.openi.org.cn/OpenI/MSAdapter.git)|msa_r0.2.0|
|vllm mindspore|[https://gitee.com/mindspore/vllm-mindspore.git](https://gitee.com/mindspore/vllm-mindspore.git)|br_infer_boom|
|golden-stick|[https://gitee.com/mindspore/golden-stick.git](https://gitee.com/mindspore/golden-stick.git)|br_infer_boom|

### 启动脚本
设置环境变量&启动ray进程:

```
export vLLM_MODEL_BACKEND=MindFormers
export MS_ENABLE_LCCL=off
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GLOO_SOCKET_IFNAME=enp125s0f0
export TP_SOCKET_IFNAME=enp125s0f0
export MS_SUBMODULE_LOG_v="{PARSER:2,ANALYZER:2}"
export DEVICE_NUM_PER_NODE=8

# 300I Duo 新增
export MS_ENABLE_INTERNAL_BOOST=off
export MS_ENABLE_TRACE_MEMORY=on
export DISABLE_SHAPE_RESAHPE=on
export MS_DEV_RUNTIME_CONF="only_local_comm:true"
export MS_ALLOC_CONF=enable_vmm:true
export MS_INTERNAL_ENABLE_CUSTOM_KERNEL_LIST=AddRmsNorm,RmsNormQuant,InferenceSwiGLU,TransposeBatchMatmulTranspose
export VLLM_PP_LAYER_PARTITION="31,30"  #pp并行层数设置

# 设置为predict_ds_671b.yaml的绝对路径
export MINDFORMERS_MODEL_CONFIG=/workspace/predict_deepseek_r1_671b_300iduo.yaml

# 主节点
ray stop
ray start --head --port=6333

#从节点
# ray stop
# ray start --address=xxx.xxx.xxx.xxx:6333  #主节点ip地址
```

启动服务化:

```
vllm-mindspore serve "/home/ckpt/deepseek_r1_671b" --trust-remote-code --no-enable-prefix-caching --dtype float16 --tensor-parallel-size=8 --pipeline_parallel_size=2 --max-num-seqs=256 --block-size=128 --gpu-memory-utilization=0.8 --max-num-batched-tokens=16384 --max-model-len=32768 --distributed-executor-backend=ray > vllm.txt 2>&1
```

参数含义可参考 [vLLM参数](https://docs.vllm.ai/en/latest/cli/index.html)

### 配置文件
模型运行需要指定yaml文件进行配置，目前镜像里已提供，可直接使用。
镜像中提供为tp8pp2配置，其余配置yaml见`https://gitee.com/mindspore/mindformers/blob/bf6586db38a81fffe65abfb89f3e743137789c86/research/deepseek3/deepseek_r1_671b`

```yaml
seed: 0
output_dir: './output' # path to save checkpoint/strategy
run_mode: 'predict'
use_parallel: True

load_checkpoint: ""
load_ckpt_format: "safetensors"
auto_trans_ckpt: False # If true, auto transform load_checkpoint to load in distributed model

# trainer config
trainer:
  type: CausalLanguageModelingTrainer
  model_name: 'DeepSeekR1-W4A8'

# default parallel of device num = 16 for Atlas 800T A2
parallel_config:
  model_parallel: 8
  pipeline_stage: 2
  expert_parallel: 1
  vocab_emb_dp: False

# mindspore context init config
context:
  mode: 1 # 0--Graph Mode; 1--Pynative Mode
  max_device_memory: "40GB"
  device_id: 0
  affinity_cpu_list: None

# parallel context config
parallel:
  parallel_mode: "STAND_ALONE" # use 'STAND_ALONE' mode for inference with parallelism in frontend
  full_batch: False
  strategy_ckpt_save_file: "./ckpt_strategy.ckpt"

# model config
model:
  model_config:
    type: DeepseekV3Config
    auto_register: deepseek3_config.DeepseekV3Config
    batch_size: 1 # add for incre predict
    seq_length: 4096
    hidden_size: 7168
    num_layers: 61
    num_heads: 128
    max_position_embeddings: 163840
    intermediate_size: 18432
    kv_lora_rank:  512
    q_lora_rank: 1536
    qk_rope_head_dim: 64
    v_head_dim: 128
    qk_nope_head_dim: 128
    vocab_size: 129280
    multiple_of: 256
    rms_norm_eps: 1.0e-6
    bos_token_id: 0
    eos_token_id: 1
    pad_token_id: 1
    ignore_token_id: -100
    compute_dtype: "float16"
    layernorm_compute_type: "float16"
    softmax_compute_type: "float16"
    rotary_dtype: "float16"
    router_dense_type: "float16"
    param_init_type: "float16"
    scaling_factor:
      beta_fast: 32.0
      beta_slow: 1.0
      factor: 40.0
      mscale: 1.0
      mscale_all_dim: 1.0
      original_max_position_embeddings: 4096
    use_past: True
    extend_method: "YARN"
    use_flash_attention: True
    block_size: 128
    num_blocks: 512
    offset: [1, 0]
    checkpoint_name_or_path: ""
    repetition_penalty: 1
    max_decode_length: 128
    top_k: 1
    top_p: 1
    theta: 10000.0
    do_sample: False
    is_dynamic: True
    qkv_concat: False
    ffn_concat: True
    quantization_config:
      quant_method: 'a8w4'
    auto_map:
      AutoConfig: deepseek3_config.DeepseekV3Config
      AutoModel: deepseek3.DeepseekV3ForCausalLM
  arch:
    type: DeepseekV3ForCausalLM
    auto_register: deepseek3.DeepseekV3ForCausalLM

moe_config:
  expert_num: 256
  num_experts_chosen: 8
  routing_policy: "TopkRouterV2"
  shared_expert_num: 1
  routed_scaling_factor: 2.5
  first_k_dense_replace: 3
  moe_intermediate_size: 2048
  topk_group: 4
  n_group: 8

processor:
  return_tensors: ms
  tokenizer:
    unk_token: '<unk>'
    bos_token: '<｜begin▁of▁sentence｜>'
    eos_token: '<｜end▁of▁sentence｜>'
    pad_token: '<｜end▁of▁sentence｜>'
    type: LlamaTokenizerFast
    vocab_file: 'path/to/tokenizer.json'
    tokenizer_file: 'path/to/tokenizer.json'
    chat_template: "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='', is_first_sp=true) %}{%- for message in messages %}{%- if message['role'] == 'system' %}{%- if ns.is_first_sp %}{% set ns.system_prompt = ns.system_prompt + message['content'] %}{% set ns.is_first_sp = false %}{%- else %}{% set ns.system_prompt = ns.system_prompt + '\n\n' + message['content'] %}{%- endif %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{{'<｜Assistant｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}"
  type: LlamaProcessor

```
