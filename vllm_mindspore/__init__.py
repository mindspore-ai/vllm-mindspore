import os
from vllm import ModelRegistry
from vllm.logger import init_logger
from mindspore import set_context

logger = init_logger(__name__)

def register_model():
    init_env()

    ModelRegistry.register_model(
        "DeepSeekMTPModel",
        "vllm_mindspore.model_executor.models.mf_models.deepseek_mtp:DeepseekV3MTPForCausalLM")

    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM",
        "vllm_mindspore.model_executor.models.mf_models.deepseek_v3:DeepseekV3ForCausalLM")

    ModelRegistry.register_model(
        "LlamaForCausalLM",
        "vllm_mindspore.model_executor.models.llama:LlamaForCausalLM")

    ModelRegistry.register_model(
        "Qwen2ForCausalLM",
        "vllm_mindspore.model_executor.models.qwen2:Qwen2ForCausalLM")

    ModelRegistry.register_model(
        "Qwen3ForCausalLM",
        "vllm_mindspore.model_executor.models.qwen3:Qwen3ForCausalLM")

    ModelRegistry.register_model(
        "Qwen2_5_VLForConditionalGeneration",
        "vllm_mindspore.model_executor.models.qwen2_5_vl:Qwen2_5_VLForConditionalGeneration")


def init_env():
    set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    defact_env = {
        "MS_ENABLE_LCCL": "off",
        "HCCL_EXEC_TIMEOUT": "7200",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "MS_JIT_MODULES": "vllm_mindspore,research",
        "RAY_CGRAPH_get_timeout": "360",
        "MS_NODE_TIMEOUT": "180",
        "MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST": "FlashAttentionScore,PagedAttention"
    }

    for key, value in defact_env.items():
        if key not in os.environ:
            logger.debug('Setting %s to "%s"', key, value)
            os.environ[key] = value
