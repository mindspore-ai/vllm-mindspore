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
import pytest
from unittest.mock import patch
import os
from multiprocessing import Process, Queue

from tests.utils.common_utils import (teardown_function, setup_function,
                                      MODEL_PATH)
from tests.utils.env_var_manager import EnvVarManager

env_manager = EnvVarManager()
env_manager.setup_mindformers_environment()
env_vars = {
    "VLLM_MS_MODEL_BACKEND": "MindFormers",
    "MS_ENABLE_LCCL": "off",
    "MS_ENABLE_TRACE_MEMORY": "off",
    "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
    "HCCL_IF_BASE_PORT": "60000",
    "LCAL_COMM_ID": "127.0.0.1:10068"
}

ds_model_path = MODEL_PATH["DeepSeek-R1-W8A8"]
common_ds_prompt = ("You are a helpful assistant.<｜User｜>将文本分类为中性、"
                    "负面或正面。 \n文本：我认为这次假期还可以。 \n情感："
                    "<｜Assistant｜>\n")
common_ds_expect_result = 'ugs611ాలు'

qwen_model_path = MODEL_PATH["Qwen3-30B-A3B"]
common_qwen_prompt = common_ds_prompt
common_qwen_expect_result = '<think>\n好的'

quant_type = 'ascend'


def dp_func(dp_size, local_dp_rank, global_dp_rank, tp_size, ep_size,
            dp_master_port, prompts, expect_list, result_q, model_path,
            quantization):
    from vllm import LLM, SamplingParams
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
    expect_list = expect_list[start:end]
    if len(prompts) == 0:
        prompts = ["Placeholder"]
    print(f"DP rank {global_dp_rank} needs to process {len(prompts)} prompts")

    sampling_params = SamplingParams(temperature=0.0,
                                     top_p=1.0,
                                     top_k=1,
                                     repetition_penalty=1.0,
                                     max_tokens=3)

    # Create an LLM.
    gpu_memory_utilization = 0.7 if model_path == ds_model_path else 0.9
    llm = LLM(model=model_path,
              tensor_parallel_size=tp_size,
              gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=4096,
              max_num_batched_tokens=8,
              max_num_seqs=8,
              trust_remote_code=True,
              enable_expert_parallel=True,
              quantization=quantization,
              additional_config={"expert_parallel": ep_size})
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"DP rank {global_dp_rank}, Prompt: {prompt!r}, "
              f"Generated text: {generated_text!r}")
        result_q.put(generated_text == expect_list[i])


def exec_model_with_dp(dp_size,
                       tp_size,
                       ep_size,
                       prompts,
                       expect_list,
                       model_path,
                       quantization=None):
    from vllm.utils import get_open_port

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
                             ep_size, dp_master_port, prompts, expect_list,
                             result_q, model_path, quantization))
        proc.start()
        procs.append(proc)
    exit_code = 0

    for proc in procs:
        proc.join(timeout=600)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that "
                  f"didn't stop within 10 minutes.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    assert exit_code == 0
    result = True
    for proc in procs:
        result = result and result_q.get()
    assert result


def exec_model_without_dp(tp_size,
                          ep_size,
                          prompts,
                          expect_list,
                          model_path,
                          quantization=None):
    from vllm import LLM, SamplingParams
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=3,
                                     top_k=1,
                                     top_p=1.0,
                                     repetition_penalty=1.0)

    # Create an LLM.
    llm = LLM(model=model_path,
              tensor_parallel_size=tp_size,
              trust_remote_code=True,
              gpu_memory_utilization=0.9,
              max_model_len=4096,
              enable_expert_parallel=True,
              quantization=quantization,
              additional_config={"expert_parallel": ep_size})
    # Generate texts from the prompts. The output is a list of
    # RequestOutput objects that contain the prompt, generated
    # text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        assert generated_text == expect_list[i]


@patch.dict(os.environ, env_vars)
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.allcards
def test_vllm_qwen3_moe_30b_dp4_tp2_ep4():
    """
    Test Summary:
        test case qwen3_moe_30B with DP4TP2EP4
    Expected Result:
        Running successfully, the first three tokens in the return result
        meet expectations.
    Model Info:
        Qwen3-30B-A3B
    """
    import vllm_mindspore

    dp_size = 4
    tp_size = 2
    ep_size = 4
    # Sample prompts.
    prompts = [common_qwen_prompt] * 4
    expect_list = [common_qwen_expect_result] * 4
    exec_model_with_dp(dp_size, tp_size, ep_size, prompts, expect_list,
                       qwen_model_path)


@patch.dict(os.environ, env_vars)
@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.allcards
def test_deepseek_r1_dp4_tp2_ep4():
    """
    Test Summary:
        test case deepseek r1 w8a8 dp4 tp2 ep4
    Expected Result:
        Running successfully, the first three tokens in the return result
        meet expectations.
    Model Info:
        DeepSeek-R1-W8A8
    """
    import vllm_mindspore

    dp_size = 4
    tp_size = 2
    ep_size = 4
    # Sample prompts.
    prompts = [common_ds_prompt] * 4
    expect_list = [common_ds_expect_result] * 4
    exec_model_with_dp(dp_size, tp_size, ep_size, prompts, expect_list,
                       ds_model_path, quant_type)


@pytest.mark.skip(
    reason=
    "Currently does not support relevant communication fusion operators in 910b"
)
@patch.dict(os.environ, env_vars)
def test_deepseek_r1_dp8_tp1_ep8():
    """
    Test Summary:
        test case deepseek r1 w8a8 Dp8 tp1 ep8
    Expected Result:
        Running successfully, the first three tokens in the return result
        meet expectations.
    Model Info:
        DeepSeek-R1-W8A8
    """
    import vllm_mindspore

    dp_size = 8
    tp_size = 1
    ep_size = 8
    # Sample prompts.
    prompts = [common_ds_prompt] * 8
    expect_list = [common_ds_expect_result] * 8
    exec_model_with_dp(dp_size, tp_size, ep_size, prompts, expect_list,
                       ds_model_path, quant_type)


@patch.dict(os.environ, env_vars)
@pytest.mark.level4
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.allcards
def test_deepseek_r1_dp2_tp4_ep1():
    """
    Test Summary:
        test case deepseek r1 w8a8 dp2 tp4 ep1
    Expected Result:
        Running successfully, the first three tokens in the return result
        meet expectations.
    Model Info:
        DeepSeek-R1-W8A8
    """
    import vllm_mindspore

    dp_size = 2
    tp_size = 4
    ep_size = 1
    # Sample prompts.
    prompts = [common_ds_prompt] * 2
    expect_list = [common_ds_expect_result] * 2
    exec_model_with_dp(dp_size, tp_size, ep_size, prompts, expect_list,
                       ds_model_path, quant_type)


@patch.dict(os.environ, env_vars)
@pytest.mark.skip(
    reason=
    "Currently does not support relevant communication fusion operators in 910b"
)
def test_deepseek_r1_dp4_tp2_ep8():
    """
    Test Summary:
        test case deepseek r1 w8a8 dp4 tp2 ep8
    Expected Result:
        Running successfully, the first three tokens in the return result
        meet expectations.
    Model Info:
        DeepSeek-R1-W8A8
    """
    import vllm_mindspore

    dp_size = 4
    tp_size = 2
    ep_size = 8
    # Sample prompts.
    prompts = [common_ds_prompt] * 4
    expect_list = [common_ds_expect_result] * 4
    exec_model_with_dp(dp_size, tp_size, ep_size, prompts, expect_list,
                       ds_model_path, quant_type)


@patch.dict(os.environ, env_vars)
@pytest.mark.level4
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.allcards
def test_deepseek_r1_dp8_tp1_ep1():
    """
    Test Summary:
        test case deepseek r1 w8a8 dp8 tp1 ep1
    Expected Result:
        Running successfully, the first three tokens in the return result
        meet expectations.
    Model Info:
        DeepSeek-R1-W8A8
    """
    import vllm_mindspore

    dp_size = 8
    tp_size = 1
    ep_size = 1
    # Sample prompts.
    prompts = [common_ds_prompt] * 8
    expect_list = [common_ds_expect_result] * 8
    exec_model_with_dp(dp_size, tp_size, ep_size, prompts, expect_list,
                       ds_model_path, quant_type)


@patch.dict(os.environ, env_vars)
@pytest.mark.level4
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.allcards
def test_deepseek_r1_dp8_tp1_ep4():
    """
    Test Summary:
        test case deepseek r1 w8a8 dp8 tp1 ep4
    Expected Result:
        Running successfully, the first three tokens in the return result
        meet expectations.
    Model Info:
        DeepSeek-R1-W8A8
    """
    import vllm_mindspore

    dp_size = 8
    tp_size = 1
    ep_size = 4
    # Sample prompts.
    prompts = [common_ds_prompt] * 8
    expect_list = [common_ds_expect_result] * 8
    exec_model_with_dp(dp_size, tp_size, ep_size, prompts, expect_list,
                       ds_model_path, quant_type)


@patch.dict(os.environ, env_vars)
@pytest.mark.level4
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.allcards
def test_deepseek_r1_tp8_ep8():
    """
    Test Summary:
        test case deepseek r1 w8a8 tp8 ep8
    Expected Result:
        Running successfully, the first three tokens in the return result
        meet expectations.
    Model Info:
        DeepSeek-R1-W8A8
    """
    import vllm_mindspore

    tp_size = 8
    ep_size = 8
    # Sample prompts.
    prompts = [common_ds_prompt]
    expect_list = [common_ds_expect_result]
    exec_model_without_dp(tp_size, ep_size, prompts, expect_list,
                          ds_model_path, quant_type)


@patch.dict(os.environ, env_vars)
@pytest.mark.level4
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.allcards
def test_deepseek_r1_tp8_ep4():
    """
    Test Summary:
        test case deepseek r1 w8a8 tp8 ep4
    Expected Result:
        Running successfully, the first three tokens in the return result
        meet expectations.
    Model Info:
        DeepSeek-R1-W8A8
    """
    import vllm_mindspore

    tp_size = 8
    ep_size = 4
    # Sample prompts.
    prompts = [common_ds_prompt]
    expect_list = [common_ds_expect_result]
    exec_model_without_dp(tp_size, ep_size, prompts, expect_list,
                          ds_model_path, quant_type)


@patch.dict(os.environ, {**env_vars, "VLLM_MS_MODEL_BACKEND": "Native"})
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.allcards
def test_vllm_native_qwen3_moe_30b_dp4_tp2_ep4():
    """
    Test Summary:
        test case qwen3_moe_30B with DP4TP2EP4
    Expected Result:
        Running successfully, the first three tokens in the return result
        meet expectations.
    Model Info:
        DeepSeek-R1-W8A8
    """
    import vllm_mindspore

    dp_size = 4
    tp_size = 2
    ep_size = 4
    # Sample prompts.
    prompts = [common_qwen_prompt] * 4
    expect_list = [common_qwen_expect_result] * 4
    exec_model_with_dp(dp_size, tp_size, ep_size, prompts, expect_list,
                       qwen_model_path)


@patch.dict(os.environ, {**env_vars, "VLLM_MS_MODEL_BACKEND": "Native"})
@pytest.mark.level4
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.allcards
def test_vllm_native_qwen3_moe_30b_tp8_ep4():
    """
    Test Summary:
        test case qwen3_moe_30B with TP8EP4
    Expected Result:
        Running successfully, the first three tokens in the return result
        meet expectations.
    Model Info:
        DeepSeek-R1-W8A8
    """
    import vllm_mindspore

    tp_size = 8
    ep_size = 4
    # Sample prompts.
    prompts = [common_qwen_prompt]
    expect_list = [common_qwen_expect_result]
    exec_model_without_dp(tp_size, ep_size, prompts, expect_list,
                          qwen_model_path, quant_type)
