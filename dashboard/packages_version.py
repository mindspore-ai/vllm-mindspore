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
import os
import site
import subprocess

import regex as re


def get_commit_info(
    app_name=None,
    pkg_dir_name=None,
    repo_path=None,
    env_prefix=None,
    short=True,
):
    """
    获取 commit id 和 branch
    1. 如果传 repo_path，则用 git 获取
    2. 否则根据 site-packages 下的 .commit_id 文件解析
    :param app_name: pip 包名 (例如 "vllm-mindspore")
    :param pkg_dir_name: site-packages 下的目录名 (例如 "vllm_mindspore")
    :param repo_path: 代码仓路径 (如果给这个，优先使用 git)
    :param env_prefix: conda/env 路径前缀 (默认自动推断)
    :param short: 是否输出短 commit id（7位）
    :return: "commit_id (branch)" 或 None
    """

    commit_id = None
    branch = None

    # ========== 1. git 仓库优先 ==========
    if repo_path:
        try:
            commit_id = (subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_path).decode().strip())
            if short:
                commit_id = commit_id[:7]

            branch = (subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_path,
            ).decode().strip())

            # 处理 detached HEAD
            if branch == "HEAD":
                try:
                    branches = (subprocess.check_output(
                        ["git", "branch", "--contains", "HEAD"],
                        cwd=repo_path,
                    ).decode().splitlines())
                    if branches:
                        branch = branches[0].replace("*", "").strip()
                    else:
                        branch = (subprocess.check_output(
                            ["git", "describe", "--tags", "--always"],
                            cwd=repo_path,
                        ).decode().strip())
                except Exception:
                    branch = "detached"
            return f"{commit_id} ({branch})"
        except Exception as e:
            print(f"Git 读取失败: {e}")
            return None

    # ========== 2. site-packages 下的 .commit_id ==========
    if env_prefix is None:
        site_packages = site.getsitepackages()[0]
    else:
        site_packages = os.path.join(
            env_prefix,
            "lib",
            f"python{os.sys.version_info.major}."
            f"{os.sys.version_info.minor}",
            "site-packages",
        )

    if not pkg_dir_name:
        raise ValueError("必须提供 pkg_dir_name 或 repo_path")

    commit_file = os.path.join(site_packages, pkg_dir_name, ".commit_id")

    if os.path.exists(commit_file):
        with open(commit_file, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        content = "\n".join(lines)

        # 提取 commit id
        sha_match = re.search(r"\b[0-9a-f]{7,40}\b", content)
        if sha_match:
            commit_id = sha_match.group(0)
            if short:
                commit_id = commit_id[:7]

        # 提取 branch
        if (len(lines) >= 2 and lines[0]
                and not re.match(r"^[0-9a-f]{7,40}$", lines[0])):
            branch = lines[0]
        else:
            branch_match = re.search(r"\[branch\]:(.*)", content)
            if branch_match:
                branch_raw = branch_match.group(1).strip("() []'")
                if "," in branch_raw:
                    branch = branch_raw.split(",")[-1].strip()
                else:
                    branch = branch_raw

    if commit_id:
        return f"{commit_id} ({branch or 'unknown'})"
    return None


if __name__ == "__main__":
    # 从 site-packages 读取
    print(
        "vllm-mindspore:",
        get_commit_info(app_name="vllm-mindspore",
                        pkg_dir_name="vllm_mindspore"),
    )
    print(
        "mindspore:",
        get_commit_info(app_name="mindspore", pkg_dir_name="mindspore"),
    )
    print(
        "mindspore_gs:",
        get_commit_info(app_name="mindspore_gs", pkg_dir_name="mindspore_gs"),
    )
    print(
        "mindformers:",
        get_commit_info(app_name="mindformers", pkg_dir_name="mindformers"),
    )
