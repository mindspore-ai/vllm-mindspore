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
'''
common utils
'''

import contextlib
import itertools
import logging
import os
import yaml
import time
import signal
import psutil
import socket
import subprocess
import random
import regex as re
import requests
import json
import jieba
import jieba.posseg as pseg
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_PATH = {}

HAS_MODEL_PATH_REGISTERED = False
# Valid port range in CI environment is [1024, 65520]
PORT_LOWER_BOUND = 1024
PORT_UPPER_BOUND = 65520
BASE_PORT = 8000
LCCL_BASE_PORT = 21000
HCCL_BASE_PORT = 41000


def register_model_path_from_yaml(yaml_file):
    """
    Register the model path to MODEL_PATH dict.
    """
    global HAS_MODEL_PATH_REGISTERED
    if not HAS_MODEL_PATH_REGISTERED:
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        module_info_yaml_path = os.path.join(current_dir, yaml_file)
        with open(module_info_yaml_path) as f:
            models = yaml.safe_load(f)

        MODEL_PATH.update({
            model_name:
            f"/home/workspace/mindspore_dataset/weight/{model_name}"
            for model_name in models
        })
        HAS_MODEL_PATH_REGISTERED = True


register_model_path_from_yaml("model_info.yaml")


def is_port_available(port):
    """Determine if the port is available (unoccupied)"""
    if not isinstance(
            port, int) or not (PORT_LOWER_BOUND <= port <= PORT_UPPER_BOUND):
        raise ValueError(
            f"The port number must be an integer between "
            f"{PORT_LOWER_BOUND}-{PORT_UPPER_BOUND}, but got {port}.")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", port))
            return True
    except OSError:
        logger.warning("Port %d is already in use", port)
        return False


def get_available_port(base_port):
    """
    Return available port numbers, first check if the base
    port is available. while not, add 10 until the port is
    available (unoccupied)
    """
    available_port = base_port
    step_offset = 10
    while not is_port_available(available_port):
        # Increment port number if already in use
        available_port += step_offset
    logger.warning("Available Port: %d", available_port)
    return available_port


def setup_function():
    """pytest will call the setup_function before case executes."""
    device_id = os.environ.pop("DEVICE_ID", None)
    if device_id is not None:
        # Specify device through environment variables to avoid the problem
        # of delayed resource release in single card cases. When executing
        # this function, DEVICE_ID has already taken effect, and the device id
        # needs to be reset to 0, otherwise it may be out of bounds.
        import mindspore as ms
        ms.set_device("Ascend", 0)
        logger.warning("This case is assigned to device:%s", str(device_id))
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device_id

    port_offset = int(device_id) if device_id else 0
    # Used to distinguish multiple online services running simultaneously.
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if not serve_port:
        serve_port = BASE_PORT + port_offset
        os.environ["TEST_SERVE_PORT"] = f"{get_available_port(serve_port)}"

    # Randomly specify LCCL and HCCL ports for cases without specified port,
    # mainly in single card concurrent scenarios, to avoid port conflicts.
    lccl_port = os.getenv("LCAL_COMM_ID", None)
    if not lccl_port:
        lccl_port = LCCL_BASE_PORT + port_offset
        os.environ[
            "LCAL_COMM_ID"] = f"127.0.0.1:{get_available_port(lccl_port)}"

    hccl_port = os.getenv("HCCL_IF_BASE_PORT", None)
    if not hccl_port:
        hccl_port = HCCL_BASE_PORT + port_offset
        os.environ["HCCL_IF_BASE_PORT"] = str(get_available_port(hccl_port))


def cleanup_subprocesses(pid=None) -> None:
    """Cleanup all subprocesses raise by main test process."""
    pid = pid if pid else os.getpid()
    cur_proc = psutil.Process(pid)
    children = cur_proc.children(recursive=True)
    for child in children:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            try:
                with contextlib.suppress(psutil.NoSuchProcess,
                                         ProcessLookupError):
                    child.kill()
            except Exception as err:
                result = subprocess.run(['ps axf'],
                                        capture_output=True,
                                        text=True,
                                        check=True)
                logger.error(
                    "Process cleanup failed, current process tree is as "
                    "follows: %s", result.stdout)
                raise err

    start_time = time.time()
    time_out = 10
    while children and (time.time() - start_time) < time_out:
        for child in list(children):
            try:
                child.wait(0.1)
                children.remove(child)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired,
                    ProcessLookupError):
                pass

    if children:
        raise RuntimeError("Resource Recycling Exception: Process not "
                           "released within 10 seconds.")


def stop_vllm_server(process=None):
    """Stop the vLLM service and its related processes."""
    pid = process.pid if process else None
    cleanup_subprocesses(pid=pid)


def teardown_function():
    """pytest will call the teardown_function after case function completed."""
    cleanup_subprocesses()


def get_key_counter_from_log(log_name, key):
    """Count keyword occurrences in the log file."""
    dirname, _ = os.path.split(os.path.abspath(__file__))
    log_path = os.path.join(dirname, log_name)
    if "'" in key:
        cmd = f"cat {log_path}|grep \"{key}\"|wc -l"
    else:
        cmd = f"cat {log_path}|grep '{key}'|wc -l"
    _, result = subprocess.getstatusoutput(cmd)
    return int(result)


def calculate_duplicate_degree(input_text, duplicate_threshold=3):
    """
    Calculate Duplication Degree
    Args:
        input_text: Input text
        duplicate_threshold: Threshold for defining duplication degree
    Returns:
        Calculated duplication degree
    """
    doc_cut = jieba.cut(input_text)
    word_counts = Counter(doc_cut)
    repeat_word_list = \
        [k for k, v in word_counts.items() if v >= duplicate_threshold]
    try:
        duplicate_degree = len(repeat_word_list) / len(word_counts)
    except ZeroDivisionError:
        duplicate_degree = 0.0
    return duplicate_degree


def calculate_garbled_degree(input_text):
    """
    Calculate Garbled Degree
    Args:
        input_text: Input text
    Returns:
        Calculated garble degree
    """
    words = pseg.cut(input_text)
    word_list = []
    flag_list = []
    for word, flag in words:
        word_list.append(word)
        flag_list.append(flag)
    flag_counts = Counter(flag_list)

    try:
        garbled_degree = flag_counts['x'] / len(flag_list)
    except ZeroDivisionError:
        garbled_degree = 0.0
    return garbled_degree


def start_vllm_server(model,
                      log_name,
                      start_mode='python3',
                      normal_case=True,
                      extra_params='',
                      error_log=None,
                      env='',
                      only_cmd=False):
    """
    Start vLLM Service Function
    Args:
        model: Model name in the request
        log_name: Name of the service startup log file
        start_mode: Startup mode, supporting "serve", "python3",
            default: "python3"
        normal_case: Whether it is a normal scenario test case, default: True
        extra_params: Additional startup parameter
        serror_log: Error keywords for service startup failures
        env: Environment variables
        only_cmd: Only return the startup command without executing it,
            default: False
    Returns:
        process: Process ID of the started service or directly return startup
            command if the only_cmd is set
    """
    dirname, _ = os.path.split(os.path.abspath(__file__))
    log_path = os.path.join(dirname, log_name)
    if start_mode == 'serve':
        start_cmd = f"vllm-mindspore serve {model}"
    elif start_mode == 'python3':
        start_cmd = "python3 -m vllm_mindspore.entrypoints " + \
                    f"vllm.entrypoints.openai.api_server --model {model}"
    else:
        raise ValueError("start_mode: wrong param!")

    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        start_cmd += f" --port={serve_port}"

    cmd = f"{env} {start_cmd} {extra_params} > {log_path} 2>&1"
    logger.info(cmd)
    if only_cmd:
        return cmd
    process = subprocess.Popen(cmd,
                               shell=True,
                               executable='/bin/bash',
                               stdout=None,
                               stderr=None,
                               preexec_fn=os.setsid)

    time.sleep(10)
    count = 0
    cycle_time = 50
    while count < cycle_time:
        result = get_key_counter_from_log(log_name,
                                          "Application startup complete")
        if result > 0:
            break
        if error_log:
            result = get_key_counter_from_log(log_name, error_log)
            if result >= 1:
                break
        result = get_key_counter_from_log(log_name, "ERROR")
        result_py = \
            get_key_counter_from_log(log_name, "temp_entrypoint.py: error")
        if result > 0 or result_py > 0:
            stop_vllm_server(process)
            if normal_case:
                with open(log_path) as f:
                    err_log = f.read()
                raise RuntimeError("vllm server fails to start!" +
                                   str(err_log))
            break
        time.sleep(10)
        count += 1
    else:
        stop_vllm_server(process)
        if normal_case:
            with open(log_path) as f:
                err_log = f.read()
            raise RuntimeError("vllm server fails to start!" + str(err_log))
    return process


def gen_random_sample_params(batch):
    """Generate random sampling parameters."""
    sample_params_list = []
    for _ in range(batch):
        post_params = {
            "temperature": random.uniform(0.0, 0.3),  # [0, 0.3]
            "max_tokens": random.randint(50, 150),  # [50, 150]
            "top_p": random.uniform(0.9, 0.95),  # [0.9, 0.95]
            "top_k": random.randint(5, 15)  # [5, 15]
        }
        sample_params_list.append(post_params)
    return sample_params_list


def send_and_get_request(data, fmt="prompt", url=None):
    """
    Send requests to specific services and capture output results.
    Args:
        data: Request data body, including models, prompt, And information
          such as sampling parameters
          e.g.  data = {"model": model,
                        "prompt": "I am",
                        "max_tokens": 100,
                        "temperature": 0}
        fmt: Supports template format, currently only supports 'chat'
          and 'prompt', default is 'prompt'
        url: URL corresponding to the service, default access
          http://localhost:8000/
    Returns:
        response: Response object, you can view the request status code
          through response.status_code or obtain the corresponding request
          text result through response.json()["choices"][0]["text"]
    """
    if not url:
        serve_port = os.getenv("TEST_SERVE_PORT", None)
        if serve_port:
            url = f'http://localhost:{serve_port}/'
        else:
            url = 'http://localhost:8000/'

    json_data = json.dumps(data)
    if fmt == "chat":
        url = url + "v1/chat/completions"
    else:
        url = url + "v1/completions"
    os.environ['NO_PROXY'] = "localhost"
    proxies = {"http": None, "https": None}
    response = requests.post(url,
                             data=json_data,
                             headers={'Content-Type': 'application/json'},
                             proxies=proxies)
    return response


def get_input_string(length, language):
    """Falsifying prompts under different configurations"""
    prompts_en = {
        5: 'What is the WTO?',
        50: 'A farmer needs to cross a river with two chickens.The boat ' + \
            'only has room for one human and two animals. What is the ' + \
            'smallest number of crossings needed for the farmer to get ' + \
            'across with the two chickens?',
        100: 'I love Beijing, because it is the capital of China and ' + \
             'there are a lot of famous places and tasty food there. ' + \
             'For example, the Great Wall, the Summer Palace and the ' + \
             'Forbidden City. As for food, Beijing duck is very ' + \
             'delicious. Beijing is a modern city as well. There are ' + \
             'many tall buildings and big supermarkets here. I can go ' + \
             'shopping in the big shopping malls. I can go to the parks ' + \
             'to do exercise. I can also go to the theatres to see ' + \
             'different shows. There are so many things to do in ' + \
             'Beijing. I love Beijing.'
    }
    prompts_ch = {
        5:
        '介绍下华为',
        50:
        '有一条绳子质地不均匀，如果从头烧到尾要花一个小时。现在有几条上述这种绳子，'
        '请你想办法计时75分钟。',
        100:
        '某日，学生小明提前放学。他独自步行15分钟后，遇到前来接送自己的父亲。随后，'
        '小明乘坐父亲驾驶的车辆返回家中。与父亲正常接送放学的时间相比，小明提前了'
        '12分钟抵达家中。请计算小明这天提前了多长时间放学。'
    }
    if length <= 100:
        if language == 'english' and length in prompts_en:
            prompt = prompts_en.get(length)
        elif language == 'chinese' and length in prompts_ch:
            prompt = prompts_ch.get(length)
        else:
            raise RuntimeError(
                f"language:{language} length:{length} prompt 不存在！")
    else:
        phrase = "Hello! " if language == 'english' else "你好!"
        repeat_times = round(length / 2)
        prompt = phrase * repeat_times
    return prompt


def process_request(model,
                    url,
                    batch,
                    concurrency,
                    fmt,
                    seq_length=100,
                    language="chinese",
                    prompt_list=None,
                    post_params=None):
    """
    Request Processing Function
    Args:
        model: Model name in the request
        url: URL corresponding to the service, default access
          http://localhost:8000/
        batch: Batch size (invalid when fmt is set to "chat")
        concurrency_levels: int, Number of concurrent tasks
        fmt: string, Supports template format, currently only supports 'chat'
          and 'prompt'
        seq_length: int, Input sequence length
        language: list, List of language types (currently supports only
          "english" and "chinese")
        prompt_list: list, Custom prompt list (batch should be set to the
          length of prompt_list in this case)
        post_params: dict, Post-processing parameters (default values:
          'temperature': 0, 'max_tokens': 100, 'top_p': 1, 'top_k': -1)
    Returns:
        list: Test results. Returns status codes if any non-200 status codes
        are encountered; duplicate results are deduplicated.
        Example return value: ["prompt output", 400]
    """
    input_str = get_input_string(seq_length, language)
    post_params = {} if post_params is None else post_params
    temperature = post_params.get('temperature', 0)
    max_tokens = post_params.get('max_tokens', 100)
    top_p = post_params.get('top_p', 1)
    top_k = post_params.get('top_k', -1)
    frequency_penalty = post_params.get('frequency_penalty', 0.0)
    presence_penalty = post_params.get('presence_penalty', 0.0)
    repetition_penalty = post_params.get('repetition_penalty', 1.05)
    logger.info(
        "Current Test: batch: %s，concurrency_levels: %s， fmt: %s，"
        "seq_length: %s，language: %s，post_params: %s， prompt_list: %s",
        str(batch), str(concurrency), fmt, str(seq_length), language,
        str(post_params), str(prompt_list))
    if fmt == "prompt":
        if prompt_list:
            prompt = prompt_list
        else:
            prompt = input_str if batch == 1 else [input_str] * batch
        data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty
        }
    elif fmt == "chat":
        message = [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": input_str
        }]
        data = {
            "model": model,
            "messages": message,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty
        }
    else:
        raise RuntimeError(f"ERROR: format only support 'prompt' and 'chat', "
                           f"not {fmt}")
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_and_get_request, data, fmt, url)
            for i in range(concurrency * 5)
        ]
        response = [future.result() for future in futures]
        # Use tqdm to monitor progress
        for _ in tqdm(as_completed(futures),
                      total=len(futures),
                      desc="Request Processing Progress"):
            pass
    result = []
    for r in response:
        if r.status_code == 200:
            if fmt == "prompt":
                for i in range(batch):
                    result.append(r.json()["choices"][i]["text"])
            if fmt == "chat":
                if r.json()["choices"][0]["message"]["content"]:
                    result.append(r.json()["choices"][0]["message"]["content"])
                if r.json()["choices"][0]["message"]["reasoning_content"]:
                    result.append(
                        r.json()["choices"][0]["message"]["reasoning_content"])
        else:
            logger.info(
                "Abnormal Request: %s, %s\nbatch: %s, concurrency: %s, "
                "data: %s", r.status_code, r.text, batch, concurrency,
                str(data)[:10])
            result.append(r.status_code)
    result = list(set(result))
    max_output_len = 5  # TODO
    if prompt_list:
        max_output_len = max(5, len(prompt_list) + 3)
    if len(result) > max_output_len:
        logger.info(
            "The output results have excessive discrepancies, "
            "with a length of %s! The data is truncated, and "
            "the length after truncation is %s!", str(len(result)),
            str(max_output_len))
        result = result[:max_output_len]
    return result


# ruff: noqa: ERA001,E501
# yapf: disable  # noqa: ERA001
def gen_mix_prompt(seq_lengths, model_max_token):
    """
    Function for Generating Mixed Dataset with Chinese, English, Numbers,
    and Emojis (Covering Multiple Batches/Sequence Lengths)
    Args:
        seq_lengths: Input sequence length
        model_max_token: Maximum token length supported by the model
    Returns:
        prompt_list: 3D array，[0]Contains prompt lists for different batches;
          [1]Contains corresponding [batch, seq_len] (sequence length for
          each batch)
    """
    batches = [1, 32]
    max_seq_len = min(max(seq_lengths), model_max_token)
    rand_num = random.randint(0, 100)
    prompt_10 = [
        "Beijing在哪里",
        "15%转换成小数是多少",
        "把abandon翻译成中文",
        "AI是什么",
        "你好，我叫Tom",  # noqa: E501
        "最新款mate的型号是",
        "用150字介绍一下华为",
        "提供一个上海游玩list"
    ]  # 8 items in total
    prompt_100 = [
        '**中国科学家成功研发6G关键技术，传输速率达1Tbps**\n\n清华大学电子工程系团队近日宣布'
        '突破6G核心技术，在太赫兹频段实现1Tbps（1000Gbps）的超高速无线传输，较当前5G速率提升'
        '100倍。\n\n该技术采用新型光子芯片，在300-500GHz频段完成实验室环境下的稳定传输测试，'
        '为6G标准制定奠定基础。项目负责人李教授表示："预计2030年前可实现商用部署。"\n\n国际电'
        '信联盟（ITU）已将该项目纳入6G标准候选技术方案。',
        '国际空间站完成首次太空番茄种植实验\nNASA宣布国际空间站成功收获首批太空种植的番茄，这是'
        'Veg-05实验项目的重要成果。宇航员使用特殊培养系统，在微重力环境下经过120天培育出5株番茄'
        '植株。实验数据显示太空番茄的维生素C含量比地面种植高出15%，为未来深空任务提供重要食物来源。',
        '英国科学家发现新型抗生素\n牛津大学研究团队在《自然》期刊发表论文，宣布发现一种名为Hos2'
        '的强效抗生素。实验表明其对耐甲氧西林金黄色葡萄球菌(MRSA)等超级细菌有效率超过95%。该抗生'
        '素从土壤稀有微生物中提取，预计2028年可进入临床使用。',
        '全球首例基因编辑治疗遗传病成功案例\n美国波士顿儿童医院为一名β地中海贫血患者实施CRISPR'
        '基因编辑治疗获得成功。治疗后6个月，患者血红蛋白水平稳定在12g/dL以上，达到正常人标准。'
        '这项突破性治疗为全球2.8万名同类患者带来希望，单次治疗费用约200万美元。',
        '德国研发新型量子计算机芯片\n马克斯·普朗克研究所成功研制出基于硅材料的量子计算芯片，在'
        '4K低温环境下实现99.9%的量子门保真度。该芯片集成16个量子比特，运算速度比传统超算快1亿'
        '倍，为可扩展量子计算机研发开辟新路径。',
        '日本建成世界最深海底实验室\n日本海洋研究开发机构在冲绳海槽建成深度达6500米的有人潜水观'
        '测站"深海6500"。该设施配备高清4K摄像系统和机械臂，可承受650个大气压，将用于研究深海生态'
        '系统和极端环境生物。',
        '波音Starliner首次载人试飞延期\n波音公司8月3日宣布Starliner飞船首次载人试飞再次延期至'
        '2024年3月。该任务原定2023年7月进行，因降落伞系统和电线胶带问题推迟。NASA已投入45亿美元'
        '支持该项目，旨在建立国际空间站第二条载人运输通道。',
        'NASA阿尔忒弥斯2号任务确定宇航员名单\nNASA于4月3日公布阿尔忒弥斯2号任务的4名宇航员名单，'
        '这是1972年以来首次载人绕月飞行任务。任务计划于2024年11月发射，使用SLS火箭和猎户座飞船，'
        '将进行为期10天的月球往返飞行。机组包括首位女性绕月宇航员Christina Koch。'
    ]  # 8 items in total
    prompt_500 = [
        'NASA宣布发现潜在宜居系外行星\n\nNASA的詹姆斯·韦伯太空望远镜近日确认了一颗编号为TOI-715 '
        'b的系外行星，位于恒星宜居带内。该行星距离地球约137光年，半径是地球的1.55倍，质量约为地球'
        '的3.8倍，公转周期19.3天。观测数据显示其表面温度可能维持在-20°C至30°C之间，具备液态水存在'
        '的可能条件。科学家通过大气光谱分析检测到水蒸气、二氧化碳和甲烷等分子特征，这是迄今为止在宜'
        '居带行星中发现的最完整大气成分数据。韦伯望远镜后续将继续观测该行星系统，计划在2024年进行更'
        '详细的大气层研究。这一发现将系外行星探索推向新高度，为寻找外星生命提供了重要目标。',
        '欧洲核子研究中心突破粒子加速技术\n\n位于瑞士的欧洲核子研究中心(CERN)近日宣布，其大型强子'
        '对撞机(LHC)升级项目取得重大进展。新型超导磁体成功将质子束能量提升至7TeV，创造了新的世界纪'
        '录。这项技术突破使得科学家能够在更小尺度上研究希格斯玻色子，预计2026年全面升级完成后，LHC'
        '的对撞能量将达到14TeV。研究人员还开发出了新型硅像素探测器，空间分辨率达到10微米，比现有设'
        '备精确5倍。这些技术进步将帮助科学家探索暗物质本质，并可能发现第五种基本作用力。该项目耗资2'
        '3亿瑞士法郎，吸引了来自85个国家的5000余名科研人员参与。',
        '全球首例猪心脏移植患者存活突破600天\n\n美国马里兰大学医学中心报告称，全球首例接受基因编辑'
        '猪心脏移植的患者David Bennett已存活超过600天，创下异种移植新纪录。这颗经过10处基因编辑的'
        '猪心脏重达320克，移植手术历时8小时完成。术后监测显示，患者体内未出现超急性排斥反应，心脏功'
        '能保持在EF值55%-60%之间。医疗团队采用了新型免疫抑制剂组合疗法，包括抗CD40抗体和雷帕霉素。'
        '这项突破为全球约1.2亿需要器官移植的患者带来希望。研究人员预计，经过进一步优化后，猪器官移植'
        '有望在5-10年内成为常规医疗手段。',
        '北极永久冻土层发现5万年前病毒仍具活性\n\n法国艾克斯-马赛大学的研究团队在西伯利亚永久冻土层'
        '中发现了一种距今约48500年的远古病毒，并在实验室成功复活。这种被命名为Pandoravirus yedoma'
        '的巨型病毒直径达1微米，含有约2500个基因。研究显示，随着全球气温上升，每年约有0.3米深的永久'
        '冻土层融化，可能释放出大量未知病原体。科学家已在冻土样本中鉴定出13种古老病毒，其中9种仍具感'
        '染性。这项发表在《病毒学杂志》的研究强调，气候变化可能带来意想不到的公共卫生风险，需要建立全'
        '球性的病原体监测网络。',
        '量子纠缠传输距离刷新至1200公里\n\n中国科学技术大学潘建伟团队利用"墨子号"量子科学实验卫星，成'
        '功实现了地面站间1200公里的量子纠缠分发。实验采用双向下行链路方案，纠缠光子对传输效率达到每秒'
        '1.6对，保真度超过80%。这项突破使构建全球量子通信网络又迈进一步。研究人员开发的新型超导纳米线'
        '单光子探测器，探测效率提升至90%，暗计数率低于1Hz。实验数据通过Quantum Key Distribution'
        '(QKD)协议加密传输，理论上可达到绝对安全。该成果发表在《自然》期刊，被认为是量子信息技术发展'
        '的重要里程碑。',
        '新型艾滋病疫苗进入III期临床试验\n\n由美国国立卫生研究院(NIH)资助的艾滋病疫苗项目取得重要进展'
        '，实验性疫苗mRNA-1644启动全球多中心III期临床试验。该疫苗基于mRNA技术，编码HIV包膜蛋白的多个'
        '保守区域。前期研究显示，在猕猴实验中保护效果达到67%，人体I/II期试验中97%的受试者产生了广谱中'
        '和抗体。本次试验将在北美、欧洲和非洲的56个中心招募3800名志愿者，预计2025年完成。如果成功，这'
        '将是首款预防HIV感染的有效疫苗，每年可防止约150万新发感染。研究团队还开发了配套的mRNA-1574疫'
        '苗作为加强针使用。',
        '南极冰芯揭示80万年前气候秘密\n\n由欧盟资助的Beyond EPICA项目团队在南极洲成功钻取到距今80万年'
        '的冰芯样本。这个长达3.2公里的冰柱包含了8个完整的冰期-间冰期旋回的气候记录。通过分析冰芯中的气泡'
        '成分，科学家发现大气二氧化碳浓度在冰期最低降至180ppm，间冰期最高达300ppm。研究还首次确认了距今'
        '约78万年前的"中更新世转型"事件，当时气候周期从4.1万年转变为10万年。这些数据将帮助改进气候模型，'
        '更准确预测全球变暖趋势。项目耗资2500万欧元，来自10个国家的32名科学家参与了这次极地考察。',
        '全球首座浮动核电站投入运营\n\n俄罗斯建造的"罗蒙诺索夫院士"号浮动核电站正式在北极港口佩韦克投入'
        '商业运行。这座144米长的海上平台配备两座35MW的KLT-40S核反应堆，可为10万人口的城镇供电。核电站采'
        '用多重安全设计，能够抵御9级地震和15米高的海浪。项目总投资达4.8亿美元，年发电量约500GWh，将取代'
        '当地老化的燃煤电厂，每年减少碳排放约20万吨。国际原子能机构(IAEA)专家已完成安全评估，认为其符合'
        '最新核安全标准。这种新型核能解决方案特别适合偏远沿海地区，已有12个国家表示引进意向。'
    ]  # 8 items in total
    prompt_1000 = [
        '这有个长新闻：国际在线专稿：近日，《巴基斯坦观察家报》网站刊发巴基斯坦地缘政治学者马哈茂德·乌'
        '尔·哈桑·汗博士（Dr. Mehmood Ul Hassan Khan）的署名评论文章：《“一带一路”与土库曼斯坦：'
        '进步与繁荣之道》。文章以不久前土库曼斯坦总统谢尔达尔·别尔德穆哈梅多夫访华、习近平主席与别尔德穆哈'
        '梅多夫总统共同宣布将中土关系提升为全面战略伙伴关系为背景，高度评价共建“一带一路”倡议为中土两国'
        '友好交往、互利共荣作出的积极贡献。\\n西方的“新大博弈”症候正在中亚地区被刻意散播以针对中国。在'
        '“一带一路”倡议发挥的关键作用影响下，中国已成为中亚地区最大贸易伙伴与投资国。\\n\\n　　作为地区'
        '内在能源、基建、交通等多个领域位居第二的大国，土库曼斯坦通过共建“一带一路”旗舰项目获得大量外商'
        '直接投资。如今，中土两国宣布提升双方关系定位，建立中土全面战略伙伴关系。自2011年起，中国一直是'
        '土库曼斯坦第一大贸易伙伴国。\\n\\n　　据中国海关2022年底数据显示，2022年1月至8月，中土双边贸'
        '易额达69亿美元，相比去年同期增长52.4%。其中，中方进口同比增长50.1%，出口同比增长87.3%。这是一'
        '个好兆头，表明随着以跨国能源项目和多模态区域运输走廊为代表的大规模基建项目的发展，土库曼斯坦正'
        '逐步向外界开放。\\n\\n　　有趣的是，土库曼斯坦地理位置优越，沟通中亚、中东、南亚与高加索地区，'
        '因而能够成为一个主要的“连接枢纽”。早在2016年，时任土库曼斯坦总统库尔班古力·米亚利克古利耶维奇·'
        '别尔德穆哈梅多夫就曾与中国国家主席习近平讨论共建“一带一路”倡议的进一步合作。\\n\\n　　土库曼斯'
        '坦是中国主要的天然气供应商之一，也是中国—中亚天然气管道项目的关键参与方。截至2022年6月，中国—中'
        '亚天然气管道在逾12年间向中国输送天然气超过4000亿立方米，折合可替代煤炭5.32亿吨，相当于减少880万'
        '吨有害物质、5.68亿吨二氧化碳气体的排放。与此同时，管道相关企业（为土库曼斯坦）提供了超过2.2万个'
        '工作岗位，培训土库曼斯坦当地员工超过11万人次。\\n\\n　　中国已成为土库曼斯坦第一大贸易伙伴国，'
        '土库曼斯坦则是中国在独联体国家中的第三大贸易伙伴。自2016年起，两国成为彼此天然气领域最大合作方。'
        '此外，土库曼斯坦是中国通往南亚、中东、东欧等地进出口市场的走廊，还连接着中国与里海。\\n\\n　　'
        '中国与土库曼斯坦于1992年正式建立外交关系，建交初期两国达成数个双边协定。近日，土库曼斯坦总统谢'
        '尔达尔·别尔德穆哈梅多夫受邀对中国进行国事访问，这是其就任总统后首次访华，访问期间签署多份合作文'
        '件及谅解备忘录，有助于未来进一步深化双边关系与共建“一带一路”倡议合作。备忘录的签署将促进数字经济'
        '、绿色发展、能源开发、医疗卫生、教育、文化、体育等领域的投资合作。\\n\\n　　在中国“一带一路”倡议'
        '启动之前，土库曼斯坦曾规划数个战略性能源基建项目和走廊，然而均受限于规模、资源及参与度。自纳入“一'
        '带一路”倡议体系后，相关项目实现升级扩张，成为洲际走廊，接入关键运输网络，获得能源安全保障。\\n\\n'
        '\\n　总体而言，两国的友好交往深化了彼此在文化、传统以及世界观层面的相互理解与互鉴。联通中土两国的'
        '伟大丝绸之路不仅仅是贸易的纽带，更架起了文化交流的桥梁。根据文章信息来看，中土合作的意义有哪些？',
        '有个新闻如下：俄媒报道称，有知情的俄罗斯官员透露，克里姆林宫已下令升级该国苏联时期的防空洞。目前'
        '俄罗斯多地的防空洞正在接受系统检查和维修。\\n据《莫斯科时报》（The Moscow Times）2月6日报道，数'
        '据统计显示，数以千计的俄罗斯地堡、防空洞和其他避难所已废弃数十年，俄罗斯缺乏足够的防空洞来容纳目前'
        '的全部人口。 俄罗斯北部城市彼得罗扎沃茨克市的官员上月表示，该市的公共防空洞只能供八分之一的该市市'
        '民容身。喀山市当局上个月表示，该市大约30%的防空洞不适合容身。一名在任的俄罗斯官员日前透露：“俄紧急'
        '情况部、国防部和（其它）文职部门已下达了对防空洞网络进行大规模检查和修缮的命令。”俄罗斯远东地区的一'
        '名政府高级官员也证实，莫斯科确实下达了升级防空洞的指令，该地区距离俄乌冲突前线有7000多公里。另据一'
        '名俄罗斯官员的说法，俄罗斯对于防空洞的检查和维修工作早在2022年2月俄乌冲突爆发伊始便已开始开展，并持'
        '续至今。\\n据报道，自2022年以来，俄罗斯政府官方门户网站上出现数百份政府招标书，用于招募公司参与全国'
        '各地的防空洞升级建设。招标项目包括通风设备维修、防水检测、空气过滤和照明设备维护等。\\n另据报道，预'
        '计今年俄罗斯克拉斯诺达尔市的地方当局将花费600多万卢布（约合人民币57.48万元）用于防空洞检修，诺夫哥'
        '罗德市将花费近5000万卢布（约合人民币479万元）用于防空洞检修，里亚赞市将花费近100万卢布（约合人民币'
        '9.58万元）。有个新闻如下：俄媒报道称，有知情的俄罗斯官员透露，克里姆林宫已下令升级该国苏联时期的防'
        '空洞。目前俄罗斯多地的防空洞正在接受系统检查和维修。\\n据《莫斯科时报》（The Moscow Times）2月6日'
        '报道，数据统计显示，数以千计的俄罗斯地堡、防空洞和其他避难所已废弃数十年，俄罗斯缺乏足够的防空洞来容'
        '纳目前的全部人口。俄罗斯北部城市彼得罗扎沃茨克市的官员上月表示，该市的公共防空洞只能供八分之一的该市'
        '市民容身。喀山市当局上个月表示，该市大约30%的防空洞不适合容身。\\n一名在任的俄罗斯官员日前透露：“俄'
        '紧急情况部、国防部和（其它）文职部门已下达了对防空洞网络进行大规模检查和修缮的命令。”俄罗斯远东地区'
        '的一名政府高级官员也证实，莫斯科确实下达了升级防空洞的指令，该地区距离俄乌冲突前线有7000多公里。另'
        '据一名俄罗斯官员的说法，俄罗斯对于防空洞的检查和维修工作早在2022年2月俄乌冲突爆发伊始便已开始开展，'
        '并持续至今。\\n据报道，自2022年以来，俄罗斯政府官方门户网站上出现数百份政府招标书，用于招募公司参与'
        '全国各地的防空洞升级建设。招标项目包括通风设备维修、防水检测、空气过滤和照明设备维护等。\\n另据报道，'
        '预计今年俄罗斯克拉斯诺达尔市的地方当局将花费600多万卢布（约合人民币57.48万元）用于防空洞检修，诺夫哥'
        '罗德市将花费近5000万卢布（约合人民币479万元）用于防空洞检修，里亚赞市将花费近100万卢布（约合人民币9.'
        '58万元）。\\n有个新闻如下：俄媒报道称，有知情的俄罗斯官员透露，克里姆林宫已下令升级该国苏联时期的防空'
        '洞。目前俄罗斯多地的防空洞正在接受系统检查和维修。\\n据《莫斯科时报》（The Moscow Times）2月6日报道'
        '，数据统计显示，数以千计的俄罗斯地堡、防空洞和其他避难所已废弃数十年，俄罗斯缺乏足够的防空洞来容纳目前'
        '的全部人口。\\n一名在任的俄罗斯官员日前透露：“俄紧急情况部、国防部和（其它）文职部门已下达了对防空洞网'
        '络进行大规模检查和修缮的命令。这篇新闻都讲了哪些重点？',
        '我这里有一篇新闻：人民网北京1月15日电 （记者赵竹青）1月14日，在农历兔年即将来临之际，中国探月航天IP形'
        '象“太空兔”的中英双语名称正式公布：中文名“兔星星”，英文名“To Star”。\\n中国探月工程亦称嫦娥工程，自200'
        '7年10月24日的嫦娥一号发射，到2019年1月3日嫦娥四号实现全人类首次月球背面软着陆和巡视探测，再到2020年12'
        '月17日嫦娥五号返回舱携带月球样品成功着陆取回1732克月壤，实现了“六战六捷”。\\n“太空兔”是中国探月航天工'
        '程的吉祥物，最初人设为中国探月派驻在月球的观察员，其形象灵感来自于中国古代神话故事中的“月宫玉兔”，结合'
        '现代宇航员形象，以红、蓝、白极具中国航天科技属性的色彩作为配色，通过拟人化的方式，让航天人奋力拼搏、勇于'
        '探索的航天精神得到了具象化体现。\\n据介绍，“兔星星”寓意“玉兔巡月，扬帆星河”，我国嫦娥三号、四号的月球车'
        '均以“玉兔”为名，“兔星星”的名字表达了其太空特质和初心使命。英文名字“To star”寓意我们的征途是星辰大海，体'
        '现了传统文化与航天科技的融合，寄托着中华民族千百年来的探月梦想，表达着新时代中国航天向宇宙深空进发的豪情'
        '愿景。我这里有一篇新闻：人民网北京1月15日电 （记者赵竹青）1月14日，在农历兔年即将来临之际，中国探月航天'
        'IP形象“太空兔”的中英双语名称正式公布：中文名“兔星星”，英文名“To Star”。\\n中国探月工程亦称嫦娥工'
        '程，自2007年10月24日的嫦娥一号发射，到2019年1月3日嫦娥四号实现全人类首次月球背面软着陆和巡视探测，再'
        '到2020年12月17日嫦娥五号返回舱携带月球样品成功着陆取回1732克月壤，实现了“六战六捷”。\\n“太空兔”是'
        '中国探月航天工程的吉祥物，最初人设为中国探月派驻在月球的观察员，其形象灵感来自于中国古代神话故事中的'
        '“月宫玉兔”，结合现代宇航员形象，以红、蓝、白极具中国航天科技属性的色彩作为配色，通过拟人化的方式，让'
        '航天人奋力拼搏、勇于探索的航天精神得到了具象化体现。\\n据介绍，“兔星星”寓意“玉兔巡月，扬帆星河”，我'
        '国嫦娥三号、四号的月球车均以“玉兔”为名，“兔星星”的名字表达了其太空特质和初心使命。英文名字“To star”'
        '寓意我们的征途是星辰大海，体现了传统文化与航天科技的融合，寄托着中华民族千百年来的探月梦想，表达着新时'
        '代中国航天向宇宙深空进发的豪情愿景。我这里有一篇新闻：人民网北京1月15日电 （记者赵竹青）1月14日，在农'
        '历兔年即将来临之际，中国探月航天IP形象“太空兔”的中英双语名称正式公布：中文名“兔星星”，英文名“To '
        'Star”。\\n\\n\\n中国探月工程亦称嫦娥工程，自2007年10月24日的嫦娥一号发射，到2019年1月3日'
        '嫦娥四号实现全人类首次月球背面软着陆和巡视探测，再到2020年12月17日嫦娥五号返回舱携带月球样品成功着陆取'
        '回1732克月壤，实现了“六战六捷”。\\n“太空兔”是中国探月航天工程的吉祥物，最初人设为中国探月派驻在月球的'
        '观察员，其形象灵感来自于中国古代神话故事中的“月宫玉兔”，结合现代宇航员形象，以红、'
        '蓝、白极具中国航天科技属性的色彩作为配色，通过拟人化的方式，让航天人奋力拼搏、勇于探索的航天精神得到了具象化'
        '体现，让航天人奋力拼搏。根据这篇文章介绍的内容，总结一下中国探月航天工程做了什么。',
        '看下这篇新闻：人民网北京1月14日电（记者孙博洋）记者从市场监管总局了解到，市场监管总局将进一步推动年报公示'
        '工作高质量发展，重点做好个体工商户年报改革，推出简易便捷的年报服务，一系列创新举措将切实为个体工商户纾困解'
        '难。\\n\\n年报登录更便捷，报送渠道更丰富。据市场监管总局相关负责人介绍，今年，个体工商户用统一社会信用代码、'
        '经营者身份证号码、工商联络员等方式均可登录国家企业信用信息公示系统。市场监管部门将年报模块嵌入微信小程序、'
        '支付宝等群众使用率较高的App，方便大家更快捷登录。\\n\\n报告事项更精简，填报设置更优化。据介绍，今年，对于'
        '行政机关已掌握的信息不再要求个体工商户填报，行政许可、特种设备情况等填报内容也无需报送。年报填报信息“一屏展'
        '示”，增加“提交并公示”提醒功能，避免漏填和未提交情形发生。\\n\\n信用修复更便利，信息同步更及时。上述相关负责'
        '人介绍，对没有在规定时间内参加年报被标记为经营异常状态的个体工商户，今年不再硬性要求到市场监管部门现场补报，'
        '允许自行网上补报，实现个体工商户年报信用修复“零跑腿”。同时，市场监管部门进一步增强国家企业信用信息公示系统的'
        '稳定性，实现年报公示信息及时同步。\\n\\n据了解，为从根本上解决现有部分个体工商户年报困难的现状，市场监管部门'
        '将对原来习惯纸质年报的个体工商户逐步建库，最终实现零录入，让数据多跑路、让群众少跑腿。\\n\\n此外，市场监管部'
        '门还将持续完善企业年报“多报合一”工作，继续实施与人力资源社会保障、商务、海关、统计、外汇部门相关事项合并年报。'
        '\\n\\n据市场监管总局相关负责人介绍，为了进一步推动年报公示工作高质量发展，市场监管总局要求各地加大政府部门数'
        '据信息归集共享力度，实现政府部门掌握的数据在企业年报时自动关联带出，提高年报数据特别是经营性相关指标的数据质'
        '量，不断降低零值率、错值率。同时，做好年报数据分析，挖掘年报数据价值。\\n\\n依法年报是每一个市场主体应尽的法'
        '定义务。市场监管总局也提示所有市场主体，应当于每年1月1日至6月30日，通过国家企业信用信息公示系统向市场监管部门'
        '报送上一年度的报告，并向社会公示。看下这篇新闻：人民网北京1月14日电（记者孙博洋）记者从市场监管总局了解到，市'
        '场监管总局将进一步推动年报公示工作高质量发展，重点做好个体工商户年报改革，推出简易便捷的年报服务，一系列创新举'
        '措将切实为个体工商户纾困解难。\\n\\n年报登录更便捷，报送渠道更丰富。据市场监管总局相关负责人介绍，今年，个体工'
        '商户用统一社会信用代码、经营者身份证号码、工商联络员等方式均可登录国家企业信用信息公示系统。市场监管部门将年报'
        '模块嵌入微信小程序、支付宝等群众使用率较高的App，方便大家更快捷登录。\\n\\n报告事项更精简，填报设置更优化。据'
        '介绍，今年，对于行政机关已掌握的信息不再要求个体工商户填报，行政许可、特种设备情况等填报内容也无需报送。年报填'
        '报信息“一屏展示”，增加“提交并公示”提醒功能，避免漏填和未提交情形发生。\\n\\n信用修复更便利，信息同步更及'
        '时。上述相关负责人介绍，对没有在规定时间内参加年报被标记为经营异常状态的个体工商户，今年不再硬性要求到市场监管部'
        '门现场补报，允许自行网上补报，实现个体工商户年报信用修复“零跑腿”。\\n\\n同时，市场监管部门进一步增强国家企业'
        '信用信息公示系统的稳定性，实现年报公示信息及时同步。\\n\\n据了解，为从根本上解决现有部分个体工商户年报困难'
        '的现状，市场监管部门将对原来习惯纸质年报的个体工商户逐步建库，最终实现零录入，让数据多跑路、让群众少跑腿。说说'
        '新闻的核心内容。',
        '根据联合国政府间气候变化专门委员会（IPCC）2023年发布的第六次评估报告（AR6），全球平均气温较工业化前'
        '水平已上升1.1℃，且以每年0.2℃的速度持续攀升。若当前碳排放趋势不变，2030年至2052年间极可能突破《巴'
        '黎协定》设定的1.5℃临界值。这一数据基于全球5000个气象观测站及NASA卫星监测结果，误差范围仅为±0.1℃。'
        '世界气象组织（WMO）秘书长Petteri Taalas在日内瓦总部发表声明称：“The window for action is '
        'closing rapidly（行动窗口正在迅速关闭）。”研究表明，升温1.5℃与2℃的差异将导致海平面上升幅度相'
        '差10厘米，直接影响全球2.8亿沿海居民，其中亚洲三角洲地区占受影响人口的70%。\n极端气候事件频率与强度'
        '的量化分析进一步印证危机迫近。2022年，欧洲热浪导致超过61,000人死亡，创下EU-MOMO死亡统计系统建立以'
        '来的峰值。同期，巴基斯坦遭遇“monsoon on steroids（强化版季风）”，降雨量达平均值的780%，造成3300'
        '万人流离失所，直接经济损失超300亿美元。美国国家海洋和大气管理局（NOAA）数据显示，2021至2023年全球发生'
        '损失超10亿美元的灾害事件共计148起，较2000至2020年平均水平上升47%。气候模型预测，若全球升温达2℃，'
        '类似2022年长江流域“热旱复合事件”的发生概率将从每50年1次增至每10年4次。应对措施的技术与经济可行性成'
        '为国际谈判焦点。国际能源署（IEA）《Net Zero by 2050》报告指出，可再生能源年投资额需从2021年的1.3'
        '万亿美元提升至2030年的4万亿美元，才能实现碳减排50%的中期目标。风能（wind power）与太阳能（solar PV）'
        '的装机容量需分别增长8倍和15倍。然而，发展中国家气候融资缺口高达5.8万亿美元，仅17%的资金流向非洲国家。'
        '剑桥大学经济政策研究所教授Diane Coyle强调：“Current financial architecture fails to address '
        'the equity dilemma（现行金融体系无法解决公平性困境）。”G20国家中，仅德国、法国达成每年1000亿美元气'
        '候援助承诺的出资比例，美国实际拨款不足承诺额的30%。\n公众意识与政策落差的矛盾日益凸显。皮尤研究中心（'
        'Pew Research Center）2023年跨国调查显示，76%的受访者认为气候变化是“major threat”，但支持碳税政策'
        '的比例降至38%。行为经济学实验表明，当个人需承担年均200美元的气候成本时，政策支持率下降22个百分点。这'
        '种“belief-action gap（信念-行动差）”在18-35岁群体中尤为显著，尽管该群体对气候危机的担忧程度高达89%，'
        '但仅有34%主动减少肉类消费或航空出行。斯坦福大学传播学教授Katharine Hayhoe指出：“Information '
        'overload has paradoxically bred paralysis（信息过载反而导致了行动瘫痪）。”',
        '根据国际可再生能源机构（IRENA）发布的《2024年可再生能源统计年鉴》，截至2023年底，全球可再生能源总装机'
        '容量达到4,004 GW，较2022年增长9.3%，首次突破4,000 GW大关。其中，太阳能光伏（solar PV）和风能（'
        'wind power）贡献了增长量的86%，分别新增装机容量295 GW和116 GW。中国、美国和欧盟继续保持领先地位，'
        '三国合计占全球新增装机量的72%。中国以惊人的178 GW新增装机量领跑全球，占全球总增量的43.5%，相当于每天'
        '新增近500 MW的清洁能源发电能力。IRENA总干事Francesco La Camera在阿布扎比总部表示：“This '
        'unprecedented growth proves that renewables are no longer alternative energy, but '
        'the backbone of the global power system（这一前所未有的增长证明，可再生能源不再是替代能源，'
        '而是全球电力系统的支柱）。”\n技术进步与成本下降是推动可再生能源爆发式增长的关键因素。自2010年以来，太'
        '阳能光伏的平准化度电成本（LCOE）下降了89%，2023年全球平均成本已降至USD 0.048/kWh，低于燃煤发电（'
        'USD 0.075/kWh）。陆上风电成本同期下降70%，达到USD 0.033/kWh。彭博新能源财经（BNEF）分析指出，在'
        '80%的国家和地区，新建太阳能或风能发电场的成本已低于运营现有化石燃料电厂。储能技术的突破进一步加速了这一'
        '趋势，2023年全球锂离子电池储能系统（BESS）部署量达到56 GWh，同比增长82%，平均成本下降至USD '
        '137/kWh。特斯拉（Tesla）Megapack等大型储能解决方案使得可再生能源的调度能力提升40%以上，有效'
        '解决了间歇性供电的难题。尽管增长迅猛，可再生能源的全球分布仍存在显著不均衡。非洲大陆虽拥有全球60%的'
        '最佳太阳能资源，但其总装机容量仅占全球的2.1%。2023年，整个撒哈拉以南非洲地区新增太阳能装机量仅为'
        '5.2 GW，不及中国一周的安装量。国际能源署（IEA）执行董事Fatih Birol警告称：“Without '
        'addressing this imbalance, the world will face a two-tier energy transition（'
        '如果不解决这种不平衡，世界将面临两级分化的能源转型）。”发展中国家面临的主要障碍包括融资成本高昂（'
        '比发达国家高7倍）、电网基础设施薄弱以及技术人才短缺。世界银行估算，到2030年，新兴市场每年需要2,10'
        '0亿美元的可再生能源投资，但目前资金缺口高达1,500亿美元。\n政策支持和市场需求的双重驱动正在重塑全'
        '球能源格局。欧盟“RepowerEU”计划承诺到2025年将可再生能源占比提高至45%，美国《通胀削减法案》（'
        'IRA）预计将带动3,690亿美元的清洁能源投资。企业采购可再生能源（corporate PPAs）在2023年创下新'
        '纪录，谷歌（Google）、亚马逊（Amazon）和微软（Microsoft）等科技巨头合计购买了38 GW的清洁电力，'
        '足够供应2,500万户家庭一年用电。市场研究机构Wood Mackenzie预测，到2030年，可再生能源将满足全球'
        '38%的电力需求，并在2040年前成为主导能源。然而，这一转型速度仍不足以实现《巴黎协定》目标，全球碳排'
        '放量在2023年仍达到36.8 Gt的创纪录水平，凸显出加速能源转型的紧迫性。\n能源转型带来的就业机会和社'
        '会影响同样值得关注。IRENA估计，2023年全球可再生能源行业直接或间接创造了1,270万个就业岗位，较2020'
        '年增长42%。其中，太阳能行业就业人数达到480万，首次超过石油和天然气开采业（450万）。然而，传统能源'
        '行业的工人面临技能转型挑战，美国劳工统计局（BLS）数据显示，2021-2023年间，约78,000名化石燃料行业'
        '工人失业，其中仅32%成功转岗至清洁能源领域。德国经济研究所（DIW）建议，各国政府应设立专项再培训基金，'
        '至少投入GDP的0.5%以确保公正转型（just transition），避免出现区域性经济衰退和社会不稳定。',
        '根据联合国人居署（UN-Habitat）最新发布的《2024年世界城市报告》，全球城市人口密度已达到每平方公里'
        '4,200人，较2010年增长37%，创下历史新高。报告涵盖全球1,934个主要城市的数据显示，亚洲城市平均密度'
        '最高，达6,800人/km²，其中孟加拉国首都达卡以惊人的47,400人/km²位居榜首，相当于每平方米居住0.047'
        '人。北美和欧洲城市密度分别为2,100人/km²和3,500人/km²，呈现明显区域差异。联合国秘书长António '
        'Guterres在报告发布会上警告："Urbanization without proper planning is creating '
        'unsustainable pressure on infrastructure（缺乏合理规划的城市化正给基础设施带来不可持续的压力'
        '）"。研究指出，全球63%的城市居民（约28亿人）将超过30%的家庭收入用于住房支出，远超世界银行设定的"可'
        '负担住房"标准（收入占比≤25%）。\n住房成本飙升与工资增长滞后的矛盾日益尖锐。美国房地产经纪人协会（'
        'NAR）数据显示，2023年全美房价中位数达到$416,000，较2019年上涨42%，而同期家庭收入中位数仅增长'
        '9.8%。伦敦政治经济学院（LSE）研究发现，在悉尼、温哥华等20个国际大都市，普通家庭需要储蓄22.3年才能'
        '支付首付款，较2000年的8.7年增长156%。这种现象催生了"沙发客一代（Generation Couchsurf）"——'
        '18-35岁人群中，39%需要与他人合租或暂住朋友家中，较2010年上升21个百分点。东京大学社会学教授田中良和'
        '指出："The traditional path of education-career-homeownership has become a privilege, '
        'not a norm（教育-职业-购房的传统路径已成为特权而非常态）"。\n城市空间不平等问题引发社会关注。卫星'
        '遥感分析显示，在全球200个特大城市中，收入前10%人群占据40%的优质居住区（指绿化率≥25%、基础设施完备的'
        '区域），而收入后50%人群挤占在仅占城市面积18%的高密度社区。巴西里约热内卢的贫富居住区平均寿命差距达12.7'
        '年，创下全球最高纪录。非政府组织"住房权利国际"（Housing Rights International）调查发现，全球有'
        '1.5亿人处于"住房不安全"状态，包括15%的纽约市民和23%的香港居民曾因租金上涨被迫搬迁。巴黎高等师范学院'
        '城市研究主任Marie Duru-Bellat强调："Gentrification is not urban renewal, but systematic '
        'exclusion（绅士化不是城市更新，而是系统性排斥）"。\n政策创新与社区自治成为破解困局的新方向。维也纳市'
        '政府的"社会住房"模式覆盖62%市民，通过限定租金为收入22%的政策，使该市连续10年位列"全球最宜居城市"。'
        '首尔市2023年推出的"共享邻里计划"（Shared Neighborhood Project）将空置办公楼改造为2,800套微型'
        '公寓（25㎡/套），租金仅为市场价60%。民间组织也在积极行动：柏林的"Kotti & Co"租户联盟成功推动租金'
        '上涨上限立法，伦敦的"Community Land Trusts"已建成1,200套永久性平价住房。世界银行城市发展局局长'
        'Sameh Wahba认为："The solution requires annual investment of 1.6trillion,butcangenerate'
        '5.4 trillion in economic benefits by 2030（解决方案需要每年1.6万亿美元投资，但到2030年可产生'
        '5.4万亿美元经济效益）"。\n人口结构变化正重塑城市需求。日本国立社会保障与人口问题研究所预测，到2040年，'
        '65岁以上老人将占东京人口的38%，催生对无障碍住房的爆发式需求。同时，全球有1.35亿"数字游民"（digital '
        'nomads）推动巴厘岛、里斯本等城市出现"工作-居住混合社区"，这类新型空间2023年增长率达240%。麦肯锡全球研'
        '究院（MGI）建议，城市规划应从"功能分区"转向"15分钟生活圈"模式，该方案已在巴黎、上海等15个城市试点，'
        '使居民通勤时间平均减少35%，碳排放降低18%。',
        '国际劳工组织（ILO）最新数据显示，2024年全球全职远程办公（remote work）劳动者占比达'
        '21%，较疫情前（2019年）的5.6%增长近3倍，另有14%采用混合办公（hybrid work）模式。'
        '美国、英国和北欧国家普及率最高，分别达到38%、32%和29%，而亚洲国家平均维持在15%左右。'
        '斯坦福大学经济政策研究所研究发现，远程工作者平均每日通勤时间减少87分钟，相当于每年节省'
        '152小时，但工作邮件往来频率上升23%，显示沟通效率面临挑战。微软（Microsoft）年度工作'
        '趋势报告指出，62%的企业已采用"3+2"模式（3天办公室+2天远程），这种安排使员工满意度提'
        '升17%，但团队协作评分下降9%。\n办公习惯变革正在深刻影响城市商业生态。高纬环球（'
        'Cushman & Wakefield）调查显示，全球主要城市写字楼空置率平均为18.7%，其中旧金山（'
        '34%）、香港（28%）和芝加哥（25%）位列前三。为应对挑战，纽约曼哈顿区已有23%的办公楼'
        '启动改造计划，包括转换为共享工作空间（co-working space）或住宅用途。星巴克（'
        'Starbucks）2023年财报披露，商务区门店销售额较2019年下降21%，而居民区门店增长13%，'
        '促使该品牌关闭156家市中心门店。伦敦政治经济学院（LSE）研究团队估算，远程办公使全球大城'
        '市中心商业区每日人流量减少4200万人次，导致相关服务业年损失达2800亿美元。\n新兴的"工作'
        '旅游"（workation）现象正在改变区域经济格局。爱彼迎（Airbnb）数据显示，2023年入住超'
        '过21天的长期住宿订单占比达38%，较2021年翻番。葡萄牙、泰国等国家推出"数字游民签证"（'
        'digital nomad visa），吸引高收入远程工作者，其中里斯本2023年新增1.2万名数字游民，'
        '带动当地房租上涨19%。日本和歌山县推出的"Workation补贴计划"已吸引4300名东京上班族短'
        '期迁移，为当地创造56亿日元经济收益。但这种现象也引发争议，巴塞罗那市民抗议"远程工作者'
        '推高房价"，促使市政府出台政策限制旅游公寓（tourist apartment）数量。\n企业人力资源'
        '管理面临范式转变。领英（LinkedIn）调查显示，83%的求职者将"工作灵活性"列为最重要择业'
        '因素，超过薪资水平（76%）。为争夺人才，亚马逊（Amazon）和谷歌（Google）等科技巨头推'
        '出"随处工作"（work from anywhere）政策，允许员工在境内自由迁徙。但德勤（Deloitte）'
        '人力资源报告指出，这种模式导致43%的企业面临跨州税务合规问题，27%的员工抱怨"永远在线"（'
        'always-on）文化加剧工作压力。麻省理工学院（MIT）斯隆管理学院建议，企业应建立"异步协'
        '作"（asynchronous collaboration）体系，通过标准化工作流程而非实时沟通提升效率，试'
        '点企业数据显示该方案可降低员工 burnout 率31%。'
    ]  # 8 items in total
    # Mixed Chinese, English, Numbers, and Emojis (1+, 10+, 100+, 500+, 1000+)
    prompt_mixed = [
        "2024年诺贝尔生理学或医学奖授予中国科学家李华教授",
        '2024年诺贝尔生理学或医学奖授予中国科学家李华教授，以表彰其在CRISPR基因编辑技术领域'
        '的突破性贡献。',
        '2024年诺贝尔生理学或医学奖授予中国科学家李华教授，以表彰其在CRISPR基因编辑技术领域'
        '的突破性贡献。李教授团队开发的"GeneClean"系统成功将基因编辑精准度提升至99.9%，为治'
        '疗5000多种遗传疾病带来新希望。颁奖典礼将于12月10日在瑞典斯德哥尔摩举行，奖金为1100'
        '万瑞典克朗（约合750万元人民币）。这是继2015年屠呦呦获奖后，中国科学家再次获得该奖项。'
        '李教授表示将把奖金全部捐献给中国青年科学家培养基金。',
        '2024年诺贝尔生理学或医学奖授予中国科学家李华教授，以表彰其在CRISPR基因编辑技术领域'
        '的突破性贡献。李教授团队开发的"GeneClean"系统成功将基因编辑精准度提升至99.9%，为治'
        '疗5000多种遗传疾病带来新希望。颁奖典礼将于12月10日在瑞典斯德哥尔摩举行，奖金为1100'
        '万瑞典克朗（约合750万元人民币）。这是继2015年屠呦呦获奖后，中国科学家再次获得该奖项。'
        '李教授表示将把奖金全部捐献给中国青年科学家培养基金。' * 3,
        '2024年诺贝尔生理学或医学奖授予中国科学家李华教授，以表彰其在CRISPR基因编辑技术领域'
        '的突破性贡献。李教授团队开发的"GeneClean"系统成功将基因编辑精准度提升至99.9%，为治'
        '疗5000多种遗传疾病带来新希望。颁奖典礼将于12月10日在瑞典斯德哥尔摩举行，奖金为1100'
        '万瑞典克朗（约合750万元人民币）。这是继2015年屠呦呦获奖后，中国科学家再次获得该奖项。'
        '李教授表示将把奖金全部捐献给中国青年科学家培养基金。' * 6
    ]
    if max_seq_len <= 10:
        prompt_mixed = [prompt_mixed[0]]
        prompt = prompt_10
        seq_len_list = [10] * 8
    elif max_seq_len <= 100:
        prompt_mixed = [prompt_mixed[1]]
        prompt = prompt_10 + prompt_100
        seq_len_list = [10] * 8 + [100] * 8
    elif max_seq_len <= 500:
        prompt_mixed = [prompt_mixed[2]]
        prompt = prompt_10 + prompt_100 + prompt_500
        seq_len_list = [10] * 8 + [100] * 8 + [500] * 8
    else:
        prompt_mixed = [prompt_mixed[3]]
        prompt = prompt_10 + prompt_100 + prompt_500 + prompt_1000
        seq_len_list = [10] * 8 + [100] * 8 + [500] * 8 + [1000] * 8

    prompt_list0 = []
    prompt_list1 = []

    for bs in batches:
        if bs == 1:
            prompt_list0.append(prompt_mixed)
            prompt_list1.append([bs, seq_len_list[-1]])
        else:
            multi_times = bs / len(prompt)
            prompt_mul = prompt * int(multi_times)
            seq_len_list_mul = seq_len_list * int(multi_times)
            bs_seq_list = []
            for seq_len in seq_len_list_mul:
                bs_seq_list.append([bs, seq_len])
            random.seed(rand_num)
            random.shuffle(prompt_mul)
            random.seed(rand_num)
            random.shuffle(bs_seq_list)
            prompt_list0.append(prompt_mul)
            prompt_list1.append(bs_seq_list)

    return [prompt_list0, prompt_list1]
# yapf: enable
# ruff: noqa


def process_offline_infer(llm,
                          batch,
                          concurrency,
                          fmt,
                          seq_length=100,
                          language="chinese",
                          prompt_list=None,
                          post_params=None):
    """
    Execute offline inference
    Args:
        llm: Initialized LLM instance
        batch: int, Batch size (invalid when fmt is set to "chat")
        concurrency_levels: int, Number of concurrent tasks
        fmt: string, Supports template format, currently only supports 'chat'
          and 'prompt'
        seq_length: int, Input sequence length
        language: list, List of language types (currently supports only
          "english" and "chinese")
        prompt_list: list, Custom prompt list (batch should be set to the
          length of prompt_list in this case)
        post_params: dict, Post-processing parameters (default values:
          'temperature': 0, 'max_tokens': 100, 'top_p': 1, 'top_k': -1)
    Returns:
        list: Test results. Returns status codes if any non-200 status codes
        are encountered; duplicate results are deduplicated.
        Example return value: ["prompt output", 400]
    """
    logger.info(
        "Current Test: batch: %s，concurrency_levels: %s， fmt: %s，"
        "seq_length: %s，language: %s，post_params: %s， prompt_list: %s",
        str(batch), str(concurrency), fmt, str(seq_length), language,
        str(post_params), str(prompt_list))
    post_params = {} if post_params is None else post_params
    temperature = post_params.get('temperature', 0)
    max_tokens = post_params.get('max_tokens', 100)
    top_p = post_params.get('top_p', 1)
    top_k = post_params.get('top_k', -1)
    from vllm import SamplingParams
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     top_p=top_p,
                                     top_k=top_k)
    input_str = get_input_string(seq_length, language)
    if fmt == "prompt":
        if prompt_list:
            prompt = prompt_list
        else:
            prompt = input_str if batch == 1 else [input_str] * batch
        llm_function = llm.generate
    elif fmt == "chat":
        prompt = [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": input_str
        }]
        llm_function = llm.chat
    else:
        raise RuntimeError(f"ERROR: format only support 'prompt' "
                           f"and 'chat', not {fmt}")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(llm_function, prompt, sampling_params)]
        outputs = [future.result() for future in futures]
    text_list = []
    for output in outputs:
        if len(output) == 0:
            text_list.append("Inference result is empty.")
        else:
            for completion_output in output[0].outputs:
                text = completion_output.text
                text_list.append(text)

    result = list(set(text_list))
    return result


def run_inference_test(is_service, model, url, llm, *args, **kwargs):
    """Execute Inference Task"""
    if is_service:
        return process_request(model, url, *args, **kwargs)
    else:
        return process_offline_infer(llm, *args, **kwargs)


def run_ac_coverage_test(is_service, model, url, llm, batches,
                         concurrency_levels, seq_lengths, formats, languages):
    """Execute AC Coverage Test"""
    param_combinations = []
    if "chat" in formats:
        chat_param_combinations = \
            itertools.product([1], concurrency_levels, ["chat"],
                              seq_lengths, languages)
        for param in chat_param_combinations:
            param_combinations.append(param)
    if "prompt" in formats:
        prompt_param_combinations = \
            itertools.product(batches, concurrency_levels, ["prompt"],
                              seq_lengths, languages)
        for param in prompt_param_combinations:
            param_combinations.append(param)

    process_results = {}
    for combination in param_combinations:
        result_comb = run_inference_test(is_service, model, url, llm,
                                         *combination)
        process_results[combination] = result_comb
    return process_results


def run_mixed_test(is_service, model, url, llm, seq_lengths, model_max_token,
                   random_post_params):
    """
    Execute Mixed Test - Covers multiple batches, multiple sequence lengths,
    and mixed Chinese-English corpora; concurrency count is 4 (online
    scenarios). Optional to add post-processing sampling parameter
    combinations.
    """
    mixed_prompt_list = gen_mix_prompt(seq_lengths, model_max_token)
    process_results = {}
    for idx, mixed_prompt in enumerate(mixed_prompt_list[0]):
        mixed_bs = len(mixed_prompt)
        post_params_list = gen_random_sample_params(batch=mixed_bs) if \
            random_post_params else [None] * mixed_bs
        concurrency = 4 if is_service else 1
        result_mixed = run_inference_test(is_service,
                                          model,
                                          url,
                                          llm,
                                          mixed_bs,
                                          concurrency,
                                          "prompt",
                                          prompt_list=mixed_prompt,
                                          post_params=post_params_list[idx])
        process_results[
            f"mixed_bs{mixed_bs}_seq{max(seq_lengths)}"] = result_mixed
    return process_results


def run_anomaly_detection_test(is_service, model, url, llm, model_max_token):
    """
    Execute Anomaly Detection Test- Intercept requests when the input exceeds
    the model's model_max_token
    """
    exceed_results = {}
    exceed_seq_length = model_max_token + 10
    if is_service:
        exceed_result = process_request(model,
                                        url,
                                        3,
                                        1,
                                        "prompt",
                                        seq_length=exceed_seq_length)

        exceed_results[f"exceed_result seq{exceed_seq_length}"] = exceed_result
        assert exceed_result[0] == 400, exceed_result[0]
    else:
        try:
            exceed_result = process_offline_infer(llm,
                                                  1,
                                                  1,
                                                  "prompt",
                                                  seq_length=exceed_seq_length)
        except ValueError as error:
            assert "is longer than the maximum model length" in str(error)
        else:
            exceed_results[
                f"exceed_result seq{exceed_seq_length}"] = exceed_result
            assert exceed_result[0] == "Inference result is empty."
            logger.info(exceed_result)


def run_combination_accuracy(model=None,
                             url=None,
                             llm=None,
                             is_service=False,
                             batches=[1, 10, 100],
                             concurrency_levels=[1, 10],
                             seq_lengths=[5, 50, 100],
                             formats=["prompt", "chat"],
                             languages=["english", "chinese"],
                             ignored_basic_check=False,
                             model_max_token=32768,
                             skip_mixed=False,
                             random_post_params=True):
    """
    Execute combination accuracy tests
    Args:
        model: Model name in the request, online service required.
        url: URL corresponding to the service, e.g. 'http://localhost:port/',
          Optional inputs for online service scenarios.
        llm: Initialized LLM instance, offline required.
        is_service: Whether enable online service inference, default is
          False (namely offline inference is enabled).
        batches: list, covered batch list; it is recommended to pass 3-5
          values.
        concurrency_levels: list, concurrency level list, it is
          recommended to pass 3-5 values. Offline inference scenarios are not
          supported temporarily.
        seq_lengths: list, input sequence length list; it is recommended to
         pass 3-5 values.
        formats: list, template format list, its items only supports 'chat'
          and 'prompt'.
        languages: list, language list, its items only supports 'english'
          and 'chinese'.
        ignored_basic_check: bool, whether to ignore basic checks (duplicate
          degree and garbled degree), default is False.
        model_max_token: Maximum token length supported by the model.
        skip_mixed: Whether to skip Part 2 mixed test, default is not to skip.
        random_post_params: Whether to set post-processing sampling parameters
          in Part 2 mixed test, default is enabled.
    Returns:
        dict: Test result statistics
        e.g.:
        {'success': 2,
         'failure': 0,
         'failed_cases': [],
         'all_cases':
             [{(1, 1, 5, 'prompt', 'english'): ['output']},
              {(1, 10 , 5, 'prompt', 'english'): ['output1', 'output2']]
         }
         where (1, 1, 5, 'prompt', 'english') serves as the dictionary key,
         corresponding to batches, concurrency_levels, seq_lengths, formats,
         and languages in order.
    """
    if is_service:
        assert model is not None, "model_name is required in online service."
        if not url:
            serve_port = os.getenv("TEST_SERVE_PORT", None)
            if serve_port:
                url = f'http://localhost:{serve_port}/'
            else:
                url = 'http://localhost:8000/'
    else:
        assert llm is not None, "llm is required in offline scenarios."

    test_results = {
        "success": 0,
        "failure": 0,
        "failed_cases": [],
        "all_cases": []
    }
    process_results = {}

    # Part 1: AC Coverage
    logger.info("Part1: AC Coverage - Generate all possible combinations "
                "among batches/concurrency/seq_len/formats")
    ac_coverage_results = run_ac_coverage_test(is_service, model, url, llm,
                                               batches, concurrency_levels,
                                               seq_lengths, formats, languages)
    process_results.update(ac_coverage_results)

    # Part 2: Mixed Test
    logger.info("Part 2: Mixed Test")
    if not skip_mixed:
        mixed_test_results = run_mixed_test(is_service, model, url, llm,
                                            seq_lengths, model_max_token,
                                            random_post_params)
        process_results.update(mixed_test_results)

    # Part 3: Anomaly Detection
    logger.info("Part3: Anomaly Detection - Intercept requests when "
                "the input exceeds the model's model_max_token")
    run_anomaly_detection_test(is_service, model, url, llm, model_max_token)

    # basic check for process_results
    for key, value in process_results.items():
        err_flag = False
        for content in value:
            # A return value is considered abnormal if it contains any error
            # code.
            if isinstance(content, int):
                logger.error("%s contains an error code: %s", key, content)
                err_flag = True
                break

            # Initially judged by duplicate degree and garbled degree, if
            # either exceeds the threshold, the result is considered invalid.
            duplicate_degree = calculate_duplicate_degree(content)
            garbled_degree = calculate_garbled_degree(content)
            if not ignored_basic_check and (duplicate_degree > 0.5
                                            or garbled_degree > 0.7):
                err_flag = True
                logger.error(
                    "%s duplicate degree and garbled degree check "
                    "failed: duplicate degree %s, garbled degree "
                    "%s.\nText content: %s", key, str(duplicate_degree),
                    str(garbled_degree), content)
                break
        if err_flag:
            test_results["failure"] += 1
            test_results["failed_cases"].append({key: value})
            test_results["all_cases"].append({key: value})
            continue
        test_results["success"] += 1
        test_results["all_cases"].append({key: value})

    return test_results


def check_hit(log_name):
    """Check whether the prefix cache is hit"""
    dirname, _ = os.path.split(os.path.abspath(__file__))
    log_path = os.path.join(dirname, log_name)
    is_hit = False
    rates = []
    with open(log_path) as f:
        log_lines = f.readlines()
    for line in log_lines:
        match = re.search(r'Prefix cache hit rate: (GPU: ){0,1}(\d+\.\d{1,})%',
                          line)
        if match:
            rates.append(float(match.group(2)))
    if rates:
        if rates[-1] > 0:
            is_hit = True
    else:
        raise ValueError("No prefix cache matched!")
    return is_hit
