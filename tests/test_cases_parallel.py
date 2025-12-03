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
"""test cases parallel"""

import os
import json
import pytest
from importlib.util import spec_from_file_location, module_from_spec

from multiprocessing.pool import Pool
from tests.utils.common_utils import (logger, teardown_function,
                                      setup_function, get_available_port,
                                      BASE_PORT, LCCL_BASE_PORT,
                                      HCCL_BASE_PORT)

level_marks = ("level0", "level1", "level2", "level3", "level4")

card_marks = ("env_onecard", "allcards", "env_single")

platform_marks = ("platform_arm_ascend910b_training", "platform_ascend310p")

PLATFORM_MAP = {
    '910B': "platform_arm_ascend910b_training",
    '310P': "platform_ascend310p"
}

HAS_TESTS_REGISTERED = False

registered_910b_tests: list = []
registered_310p_tests: list = []


def reset_registered_list():
    registered_910b_tests.clear()
    registered_310p_tests.clear()


def register_tests_by_platform(register_cases, register_list):
    """
    Register function for specific platform
    """
    for test_case in register_cases:
        """
        card_num: number of occupied cards.
        test_node_id: string in {test_file_path}::{test_function_name} format.
        """
        card_num = test_case.get("card_num")
        test_node_id = test_case.get("test_node_id")
        if card_num is not None and test_node_id is not None:
            register_list.append((card_num, test_node_id))
        else:
            logger.warning("Invalid test case entry: %s", test_case)


def load_registered_tests_from_json(json_file):
    """
    Register the tests to registered_910b_tests and registered_310p_tests.
    """
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    register_json_path = os.path.join(current_dir, json_file)
    with open(register_json_path) as f:
        tests_cases = json.load(f)

    register_tests_by_platform(tests_cases.get("registered_910b_tests"),
                               register_list=registered_910b_tests)
    register_tests_by_platform(tests_cases.get("registered_310p_tests"),
                               register_list=registered_310p_tests)


def tasks_resource_alloc(tasks: list[tuple[int]]) -> list[tuple[str]]:
    """
    Allocate devices, lccl base port, hccl base port to tasks
    according to device requirement of each task.

    For example:
        [(2, "cases_parallel/vllm_task.py::test_1", "test_1.log")]
        ==> [("export ASCEND_RT_VISIBLE_DEVICES=0,1 &&
               export LCAL_COMM_ID=127.0.0.1:10068 && "
              "export HCCL_IF_BASE_PORT=61000 && "
              "pytest -s -v cases_parallel/vllm_task.py::test_1 > test_1.log",
              "test_1.log")]

    Args:
        tasks (list[tuple[int]]): list of tasks. Each task contain 3 elements.
            1. device_req (int): Num of device requirements,
                                 which will occur device_req devices,
                                 device_req ports for lccl,
                                 device_req ports for hccl.
            2. case_desc (str): The case description,
               such as "path_to_case/case.py::target_case".
            3. log_file (str): The logging file path.

    Returns:
        list[tuple[str]]: Append resource environment to the task commands.
    """
    device_limit = 8
    device_base = 0
    lccl_base_port = LCCL_BASE_PORT
    hccl_base_port = HCCL_BASE_PORT
    base_port = BASE_PORT

    out_tasks: list[tuple[str]] = []
    for task in tasks:
        assert len(task) == 3
        resource_req, task_case, log_file = task
        if not isinstance(resource_req, int):
            raise TypeError(
                "First argument of task should be a int or str, but got %s!",
                str(type(resource_req)))

        device_str = ",".join(
            [str(d) for d in range(device_base, device_base + resource_req)])
        lccl_str = f"127.0.0.1:{get_available_port(lccl_base_port)}"

        commands = [
            f"export ASCEND_RT_VISIBLE_DEVICES={device_str}",
            f"export LCAL_COMM_ID={lccl_str}",
            f"export HCCL_IF_BASE_PORT={get_available_port(hccl_base_port)}",
            f"export TEST_SERVE_PORT={get_available_port(base_port)}"
        ]

        device_base += resource_req
        lccl_base_port += resource_req
        hccl_base_port += resource_req
        base_port += resource_req

        commands.append(f"pytest -s -v {task_case} > {log_file}")
        out_tasks.append((" && ".join(commands), log_file))

    if device_base > device_limit:
        raise ValueError(
            "Total require device %d exceeding resource limits %d !",
            device_base, device_limit)

    return out_tasks


def generate_group_contents(tests_info, capacity=8):
    """
    Group and combine the registered tests according to the given rule,
    which prioritizes those occupied more cards. Strive to maximize the
    utilization of device capacity.
    Args:
        tests_info (list): Each element in the list includes `card_num` and
            `test_node_id`
        capacity (int): The capacity of cards in the execution device,
            default 8
    Returns:
        list[list]: Tests groups divided by sorting strategy and device
            capacity, each group containing information about `card_num`
            and `test_node_id`involved
    """
    # Sort by the number of occupied devices in descending order.
    tests_info_sorted = sorted(tests_info, key=lambda x: x[0], reverse=True)
    groups: list = []  # The total number of cards occupied by each group.
    group_contents: list = []  # Store test information for each group.

    for info in tests_info_sorted:
        num = info[0]
        # Check if there are any existing groups that can accommodate the
        # current number.
        found = False
        for i in range(len(groups)):
            if groups[i] + num <= capacity:
                groups[i] += num
                group_contents[i].append(info)
                found = True
                break
        # If no feasible group is found, create a new group.
        if not found:
            groups.append(num)
            group_contents.append([info])

    return group_contents


def get_module_pytest_marks(module_path, function_name):
    """Obtain the pytestmark of the test module."""
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    module_file_path = os.path.join(current_dir, *module_path.split('/'))
    if not os.path.exists(module_file_path):
        raise ImportError("module file %s does not exist.", module_file_path)

    spec = spec_from_file_location(module_path, module_file_path)
    if spec is None:
        raise ImportError("Load module from file failed")
    module = module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as e:
        raise ImportError("Loading module %s failed: %s", module_path,
                          str(e)) from e

    func = getattr(module, function_name, None)
    if func is None:
        raise AttributeError("Function %s not found in module %s.",
                             function_name, module_path)

    if hasattr(func, 'pytestmark'):
        return [mark.name for mark in func.pytestmark]
    else:
        return []


def collection_cases_by_level(test_register):
    '''
    According to the level of registered tests, divide them into
    corresponding maps.
    '''
    tests_info: dict = {
        f"{level_marks[0]}": [],
        f"{level_marks[1]}": [],
        f"{level_marks[2]}": [],
        f"{level_marks[3]}": [],
        f"{level_marks[4]}": []
    }

    for case in test_register:
        module_path, test_function_name = case[1].split("::")
        marks = get_module_pytest_marks(module_path, test_function_name)
        level_mark = [mark for mark in marks if mark in level_marks]
        card_mark = [mark for mark in marks if mark in card_marks]

        if len(card_mark) > 0:
            raise ValueError(
                "If the case has specified 'env_single', 'env_onecard' "
                "or 'allcards', there is no need to register and execute "
                "concurrently")
        elif len(level_mark) > 1:
            raise ValueError(
                "Each test case can only specify a unique level, "
                "but %s got %s.", case[1], str(len(level_mark)))
        elif len(level_mark) == 1:
            tests_info[level_mark[0]].append(case)
        else:
            raise ValueError(
                "Case '%s' lacks necessary level mark, "
                "please specify", case[1])
    return tests_info


def generate_parallel_cases(test_register, platform):
    """
    Generate composite concurrent tests content for all registered tests.
    Args:
        test_register (list): Registration List, option from
            `registered_910b_tests` and `registered_310p_tests`
        platform (str): Corresponding platform, option from `910B` and `310P`
    """
    tests_info = collection_cases_by_level(test_register)

    for level in level_marks:
        generate_cases_with_level(tests_info[level], platform, level=level)


def generate_cases_with_level(tests_info, platform, level="level0"):
    """
    Generate composite concurrent tests content based on the specified
    test level.
    Args:
        tests_info (list): Tests information at the corresponding level
        platform (str): Corresponding platform, option from `910B` and `310P`
        level (str): Tests level, supporting levels 0 to 4
    """
    if len(tests_info) == 0:
        return

    group_contents = generate_group_contents(tests_info, capacity=8)

    for i, per_group in enumerate(group_contents):
        print(f"iter: {i}. per_group: {per_group}\n")
        test_content = ""
        test_content += (
            f"@pytest.mark.{level}\n"
            f"@pytest.mark.{PLATFORM_MAP[platform]}\n"
            f"@pytest.mark.env_single\n"
            f"def test_cases_parallel_{platform}_{level}_part{i}():\n"
            f"    cases = [\n")

        for case in per_group:
            node_id = case[1]
            log_name = node_id.split('/')[-1].replace(".py::", '_') + '.log'
            test_content += f"        ({case[0]}, '{node_id}', '{log_name}'),\n"

        test_content += ("    ]\n"
                         "    run_tasks(cases)\n\n\n")

        exec(test_content, globals())


def run_command(command_info):
    cmd, log_path = command_info
    ret = os.system(cmd)
    return ret, log_path


def check_results(commands, results):
    error_idx = [_ for _ in range(len(results)) if results[_][0] != 0]
    for idx in error_idx:
        print(f"testcase {commands[idx]} failed. "
              f"Please check log {results[idx][1]}.")
        os.system(f"grep -E 'ERROR|error|Error' {results[idx][1]} -C 5")
    assert error_idx == []


def run_tasks(cases):
    commands = tasks_resource_alloc(cases)

    with Pool(len(commands)) as pool:
        results = list(pool.imap(run_command, commands))
    check_results(commands, results)


def retrieve_tests_from_path(abs_path):
    """Retrieve tests from the specified path (ut/st)"""
    if os.path.exists(abs_path):
        logger.warning(
            "Collect and execute parallel tests under the directory: %s.",
            abs_path)
        reset_registered_list()
        register_json_path = os.path.join(abs_path,
                                          "register_parallel_tests.json")
        load_registered_tests_from_json(register_json_path)

        # Dynamically generate test cases
        generate_parallel_cases(registered_910b_tests, platform="910B")
        generate_parallel_cases(registered_310p_tests, platform="310P")


def load_and_generate_tests():
    """
    Load and generate tests form register_parallel_tests.json
    """
    current_abs_path = os.path.abspath(
        os.path.join(os.path.abspath(__file__), ".."))
    st_abs_path = os.path.join(current_abs_path, "st")
    ut_abs_path = os.path.join(current_abs_path, "ut")
    if not os.path.exists(st_abs_path) and not os.path.exists(ut_abs_path):
        raise RuntimeError("Invalid path for register_parallel_tests.json")

    global HAS_TESTS_REGISTERED
    if not HAS_TESTS_REGISTERED:
        retrieve_tests_from_path(ut_abs_path)
        retrieve_tests_from_path(st_abs_path)
        HAS_TESTS_REGISTERED = True


load_and_generate_tests()
