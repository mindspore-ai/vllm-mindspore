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
import subprocess
import sys
import time

import numpy as np


def exec_shell_cmd(cmd, execute_times=1, return_type=True, user_input=None):
    if not isinstance(cmd, str):
        print(f"TypeError: cmd type: {type(cmd)} must be str")
        return False
    for times in range(execute_times):
        sub = subprocess.Popen(args=cmd,
                               shell=True,
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
        if user_input is None:
            stdout_data, stderr_data = sub.communicate()
        else:
            stdout_data, stderr_data = sub.communicate(input=user_input)
        if sub.returncode == 0:
            print(f"Exec [{cmd}] success")
            return stdout_data.strip()
        elif sub.returncode != 0 and stderr_data:
            print(f"Exec [{cmd}] fail, status:[{sub.returncode}], "
                  f"stderr:[{stderr_data}], times:{times + 1}, "
                  f"stdout:[{stdout_data.strip()}]")
            if not return_type:
                return stderr_data.strip()
        elif sub.returncode != 0 and stdout_data and not return_type:
            return stdout_data.strip()
        else:
            print(f"Exec [{cmd}] success, status:[{sub.returncode}], "
                  f"no return, times:{times + 1}, "
                  f"stdout:[{stdout_data.strip()}]")
    return False


def get_status_cmd(cmd):
    status, value = subprocess.getstatusoutput(cmd)
    if int(status) != 0:
        print(f"Exec [{cmd}] fail, status:[{status}] return: [{value}]")
        return False
    print(f"Exec [{cmd}] success")
    return True


def shell_sed_cmd(path, old_list, new_list, file, mark_flag=False):
    print(f"shell_sed_process: {path}/{file}")
    if len(old_list) != len(new_list):
        print(
            f"Old list {old_list} len != new list {new_list} len, please check."
        )
        return False
    for i in range(len(old_list)):
        if mark_flag:
            cmd = (f"cd {path};sed -i \"s#{old_list[i]}#"
                   f"{new_list[i]}#g\" {file}")
        else:
            cmd = (f"cd {path};sed -i 's#{old_list[i]}#"
                   f"{new_list[i]}#g' {file}")
        get_status_cmd(cmd)
        print(f"sed success: [{old_list[i]} -> {new_list[i]}], file: {file}")
    return True


def check_benchmark_acc(**kwargs):
    accuracy = kwargs.get("accuracy")
    acc_stand = kwargs.get("acc_stand")
    acc_error = kwargs.get("acc_error", 1)
    acc_standard_error = np.round(float(acc_stand) * acc_error, 2)
    if accuracy < acc_standard_error:
        print(f"accuracy:{accuracy} < {acc_standard_error}")
        return False
    print(f"accuracy:{accuracy} >= {acc_standard_error}")
    return True


def aisbench_env_deployment(aisbench_source):
    cmd_list = []
    cmd_list.append(f"python -c 'import ais_bench' 2>/dev/null || ("
                    f"cd {aisbench_source}/benchmark;"
                    f"pip install -e ./;"
                    f"pip install -r requirements/api.txt)")
    return get_status_cmd(';'.join(cmd_list))


def exec_cmd(*args, **kwargs):
    cmd = args[0]
    script_path = kwargs.get("script_path")
    script_exec_mode = kwargs.get("script_exec_mode", "script")
    timeout = kwargs.get("timeout", 1000)

    if script_exec_mode == "script":
        if script_path is None:
            print("ValueError: missing script_path")
            return False
        get_status_cmd(cmd)
    else:
        sub = subprocess.Popen(args=cmd, shell=True)
        _, stderr_data = sub.communicate(timeout=timeout)
        if sub.returncode != 0:
            print(f"Exec [{cmd}] fail, status:[{sub.returncode}], "
                  f"stderr:[{stderr_data}]")
            return False
    return True


def process_check(cycle_time, cmd, wait_time=1, exec_mode="aisbench"):
    for i in range(cycle_time):
        time.sleep(wait_time)
        result = exec_shell_cmd(cmd)
        print(f"times: {i}, process num is: {int(result)}", flush=True)
        if int(result) != 0:
            if int(result) > 0:
                print(f"do {exec_mode}", flush=True)
                sys.stdout.flush()
                continue
        else:
            print(f"{exec_mode} over", flush=True)
            sys.stdout.flush()
            break
    else:
        print(f"Process timeout, max: {int(cycle_time) * int(wait_time)}",
              flush=True)
        return False
    return True


def aisbench_test(aisbench_source, models, datasets, **kwargs):
    path = kwargs.get('path')
    model = kwargs.get('model')
    host_ip = kwargs.get('host_ip', '0.0.0.0')
    host_port = kwargs.get('host_port', 8000)
    max_out_len = kwargs.get('max_out_len', 2048)
    batch_size = kwargs.get('batch_size', 256)
    wait_time = kwargs.get('wait_time', 200)

    if aisbench_env_deployment(aisbench_source):
        old_list = [
            "path=.*",
            "model=.*",
            "host_ip.*",
            "host_port.*",
            "max_out_len.*",
            "batch_size.*",
            "temperature.*",
        ]
        new_list = [
            f"path='{path}',",
            f"model='{model}',",
            f"host_ip = '{host_ip}',",
            f"host_port = {host_port},",
            f"max_out_len = {max_out_len},",
            f"batch_size = {batch_size},",
            "temperature = 0.0,",
        ]
        cfg_file = (f"benchmark/ais_bench/benchmark/configs/"
                    f"models/vllm_api/{models}.py")
        if not shell_sed_cmd(
                aisbench_source, old_list, new_list, cfg_file, mark_flag=True):
            print("set config failed")
            return False

        benchmark_cmd = ("unset http_proxy https_proxy;unset USE_TORCH;"
                         f"cd {aisbench_source}/benchmark;"
                         f"ais_bench --models {models} --datasets {datasets} "
                         f"--work-dir={path} --merge-ds --debug "
                         f"> {path}/{datasets}_bench.log 2>&1 &")
        print(benchmark_cmd)
        exec_cmd(benchmark_cmd, script_exec_mode="python")

        cmd = ("ps -ef | grep ais_bench | grep -v grep | "
               "grep -v nose | wc -l")
        process_check(720, cmd, wait_time, exec_mode="aisbench")

        res = exec_shell_cmd(
            f"grep 'accuracy' {path}/{datasets}_bench.log | "
            "head -n 1 | awk '{print $NF}' | awk -F '}' '{print $1}'")
        try:
            print(model, "'s", datasets, "result is:", float(res.strip()))
            return round(float(res.strip()), 2)
        except Exception as e:
            print(f'Benchmark test failed, {e}')
    return False
