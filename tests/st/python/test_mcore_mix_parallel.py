# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# isort:skip_file
"""test vllm mix parallel."""
import os
from multiprocessing import Process, Queue

import pytest

from . import utils
from .utils import cleanup_subprocesses


def teardown_function():
    cleanup_subprocesses()


env_manager = utils.EnvVarManager()
env_vars = {
    "ASCEND_CUSTOM_PATH": os.path.expandvars("$ASCEND_HOME_PATH/../"),
    "vLLM_MODEL_BACKEND": "MindFormers",
    "MS_ENABLE_LCCL": "off",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    "MS_ALLOC_CONF": "enable_vmm:True",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
    "HCCL_IF_BASE_PORT": "60000",
    "LCAL_COMM_ID": "127.0.0.1:10068"
}
env_manager.setup_ai_environment(env_vars)
import vllm_mindspore  # noqa: F401, E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.utils import get_open_port  # noqa: E402


def dp_func(dp_size, local_dp_rank, global_dp_rank, tp_size, ep_size,
            dp_master_port, prompts, except_list, result_q, model_path):
    dp_master_ip = "127.0.0.1"

    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    promts_per_rank = len(prompts) // dp_size
    start = global_dp_rank * promts_per_rank
    end = start + promts_per_rank
    prompts = prompts[start:end]
    except_list = except_list[start:end]
    if len(prompts) == 0:
        prompts = ["Placeholder"]
    print(f"DP rank {global_dp_rank} needs to process {len(prompts)} prompts")

    sampling_params = SamplingParams(temperature=0.0,
                                     top_p=1.0,
                                     top_k=1,
                                     repetition_penalty=1.0,
                                     max_tokens=3)

    # Create an LLM.
    llm = LLM(model=model_path,
              tensor_parallel_size=tp_size,
              max_model_len=4096,
              max_num_batched_tokens=8,
              max_num_seqs=8,
              trust_remote_code=True,
              enable_expert_parallel=True,
              additional_config={"expert_parallel": ep_size})
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"DP rank {global_dp_rank}, Prompt: {prompt!r}, "
              f"Generated text: {generated_text!r}")
        result_q.put(generated_text == except_list[i])


def exec_model_with_dp(dp_size, tp_size, ep_size, prompts, except_list,
                       model_path):
    node_size = 1
    node_rank = 0
    dp_master_port = get_open_port()
    dp_per_node = dp_size // node_size

    result_q = Queue()  # type: Queue[bool]
    procs = []
    for local_dp_rank, global_dp_rank in enumerate(
            range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)):
        proc = Process(target=dp_func,
                       args=(dp_size, local_dp_rank, global_dp_rank, tp_size,
                             ep_size, dp_master_port, prompts, except_list,
                             result_q, model_path))
        proc.start()
        procs.append(proc)
    exit_code = 0

    for proc in procs:
        proc.join(timeout=300)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that "
                  f"didn't stop within 3 minutes.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    assert exit_code == 0
    result = True
    for proc in procs:
        result = result and result_q.get()
    assert result

    # unset env
    env_manager.unset_all()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.allcards
def test_vllm_qwen3_moe_30b_dp4_tp2_ep4():
    """
    test case qwen3_moe_30B with DP4TP2EP4
    """
    dp_size = 4
    tp_size = 2
    ep_size = 4
    # Sample prompts.
    prompts = [
        "You are a helpful assistant.<｜User｜>将文本分类为中性、负面或正面。 "
        "\n文本：我认为这次假期还可以。 \n情感：<｜Assistant｜>\n",
    ] * 4
    except_list = ['<think>\n好的'] * 4
    model_path = "/home/workspace/mindspore_dataset/weight/Qwen3-30B-A3B"

    exec_model_with_dp(dp_size, tp_size, ep_size, prompts, except_list,
                       model_path)
