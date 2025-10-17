import os
from vllm import ModelRegistry
from vllm.logger import init_logger

logger = init_logger("vllm_mindspore.models")

def register_model():
    init_env()
    init_context()

    ModelRegistry.register_model(
        "Qwen2ForCausalLM",
        "vllm_mindspore.model_executor.models.qwen2:Qwen2ForCausalLM")

    # ms_base_mode = "vllm_mindspore.external.common_model:MindSporeForCausalLM"
    # ModelRegistry.register_model("Qwen2ForCausalLM", ms_base_mode)
    # ModelRegistry.register_model("Qwen3ForCausalLM", ms_base_mode)
    # ModelRegistry.register_model("LlamaForCausalLM", ms_base_mode)


def init_env():
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
    
def init_context():
    from mindspore import set_context
    set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
