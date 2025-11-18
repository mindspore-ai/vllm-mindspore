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
import argparse
import os
import shutil
import subprocess
import sys
from urllib.parse import urljoin

import regex as re
import requests
import urllib3
from bs4 import BeautifulSoup
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
"""
usage:
    python update_package.py mindspore --no-cleanup --download-only
    python update_package.py mindformers --no-cleanup --download-only
    python update_package.py golden-stick --no-cleanup --download-only \
        --whl-keyword mindspore_gs
    python update_package.py msadapter --no-cleanup --download-only
    python update_package.py vllm-mindspore --no-cleanup --download-only \
        --whl-keyword vllm_mindspore
with date/commit id:
    python update_package.py mindformers --no-cleanup --download-only \
        --date=20250912 --commit-id=e034ea
"""

BASE_DOWNLOAD_DIR = "/tmp/package_downloads"


def get_latest_month_url(base_root):
    resp = requests.get(base_root, verify=False)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    month_dirs = sorted([
        a["href"] for a in soup.find_all("a", href=True)
        if a["href"].strip("/").isdigit() and len(a["href"].strip("/")) == 6
    ])
    if not month_dirs:
        raise RuntimeError("未找到月份目录")
    return urljoin(base_root, month_dirs[-1])


def get_latest_day_url(month_url, specified_day=None):
    resp = requests.get(month_url, verify=False)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    day_dirs = sorted([
        a["href"] for a in soup.find_all("a", href=True)
        if a["href"].strip("/").isdigit() and len(a["href"].strip("/")) == 8
    ])
    if not day_dirs:
        raise RuntimeError(f"未找到日期目录: {month_url}")

    if specified_day:
        day_str = specified_day
        if f"{day_str}/" not in day_dirs:
            raise RuntimeError(f"指定日期 {day_str} 不存在于 {month_url}")
        return urljoin(month_url, f"{day_str}/"), day_str

    return urljoin(month_url, day_dirs[-1]), day_dirs[-1]


def get_branch_url(day_url,
                   branch_keyword="master",
                   commit_id=None,
                   fixed=False):
    resp = requests.get(day_url, verify=False)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    if commit_id:
        dirs = [
            a["href"] for a in soup.find_all("a", href=True)
            if commit_id in a["href"]
        ]
        if not dirs:
            raise RuntimeError(f"未找到 commit_id={commit_id} 目录: {day_url}")
        return urljoin(day_url, dirs[0])

    if fixed:
        candidate_dirs = [
            a["href"] for a in soup.find_all("a", href=True)
            if branch_keyword.lower() in a["href"].lower()
        ]
        if not candidate_dirs:
            raise RuntimeError(f"未找到匹配 {branch_keyword} 的目录: {day_url}")
        return urljoin(day_url, sorted(candidate_dirs)[-1])
    else:
        candidate_dirs = [
            a["href"] for a in soup.find_all("a", href=True)
            if branch_keyword.lower() in a["href"].lower()
        ]
        if not candidate_dirs:
            raise RuntimeError(f"未找到匹配 {branch_keyword} 的目录: {day_url}")
        return urljoin(day_url, sorted(candidate_dirs)[-1])


def extract_commit_id(fixed_path, branch_keyword, branch_url, whl_url=None):
    if fixed_path:
        match = re.search(branch_keyword + r'_\d+_([0-9a-f]+)', branch_url,
                          re.IGNORECASE)
    else:
        match = re.search(branch_keyword + r'_\d+_([0-9a-f]+)_newest',
                          branch_url, re.IGNORECASE)
    if match:
        return match.group(1)[:8]
    if whl_url:
        match_whl = re.search(r'([0-9a-f]{8,40})', whl_url)
        if match_whl:
            return match_whl.group(1)[:8]
    return "unknown"


def find_whl_by_subdirs(master_url, py_major, py_minor, whl_name_keyword=None):
    subdirs = ["any/", "unified/aarch64/", "ascend/aarch64/"]
    whl_files = []
    py_tag = f"py{py_major}"
    py_tag_full = f"cp{py_major}{py_minor}"

    for sub in subdirs:
        url = urljoin(master_url, sub)
        try:
            resp = requests.get(url, verify=False)
            resp.raise_for_status()
        except requests.HTTPError:
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if (href.endswith(".whl") and
                (py_tag in href.lower() or py_tag_full in href.lower())) \
                and (whl_name_keyword is None
                        or whl_name_keyword.lower() in href.lower()):
                whl_files.append(urljoin(url, href))
        if whl_files:
            break

    if not whl_files:
        raise RuntimeError(f"未找到符合条件whl，Python {py_major}.{py_minor}")

    whl_files.sort()
    return whl_files[-1]


def cleanup_old_packages():
    if os.path.exists(BASE_DOWNLOAD_DIR):
        shutil.rmtree(BASE_DOWNLOAD_DIR)


def download_package(url, date_str, commit_id, package_name):
    dir_name = f"{package_name}_{commit_id}"
    download_dir = os.path.join(BASE_DOWNLOAD_DIR, date_str, dir_name)
    os.makedirs(download_dir, exist_ok=True)
    filename = url.split("/")[-1]
    filepath = os.path.join(download_dir, filename)

    if os.path.exists(filepath):
        print(f"已存在 {filepath}，跳过下载")
        return filepath

    print(f"下载 {url} 到 {filepath}")
    with requests.get(url, stream=True, verify=False) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(filepath, "wb") as f, tqdm(total=total,
                                             unit='B',
                                             unit_scale=True,
                                             desc="下载中") as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return filepath


def install_package(filepath):
    print(f"安装 {filepath} ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--no-deps",
        "--force-reinstall", filepath
    ])


def update_package(package_name,
                   whl_name_keyword=None,
                   branch_keyword="master",
                   download_only=False,
                   cleanup_old=True,
                   specified_day=None,
                   commit_id=None):
    if cleanup_old:
        cleanup_old_packages()

    base_root = (
        f"https://repo.mindspore.cn/mindspore/{package_name}/version/")
    py_major, py_minor = sys.version_info.major, sys.version_info.minor

    month_url = get_latest_month_url(base_root)
    day_url, day_str = get_latest_day_url(month_url, specified_day)
    print(f"最新日期目录: {day_url}")

    fixed_path = package_name.lower() == "mindspore"
    branch_url = get_branch_url(day_url,
                                branch_keyword=branch_keyword,
                                commit_id=commit_id,
                                fixed=fixed_path)
    print(f"最新 {branch_keyword} 目录: {branch_url}")

    pkg_url = find_whl_by_subdirs(branch_url, py_major, py_minor,
                                  whl_name_keyword)
    commit_id = commit_id or extract_commit_id(fixed_path, branch_keyword,
                                               branch_url, pkg_url)
    print(f"使用 commit_id: {commit_id}")

    filepath = download_package(pkg_url, day_str, commit_id, package_name)
    if not download_only:
        install_package(filepath)
    else:
        print(f"已下载到 {filepath}，未安装")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package_name", help="组件名")
    parser.add_argument("--whl-keyword", help="whl关键字(组件名可能与whl不同)")
    parser.add_argument("--branch",
                        default="master",
                        help="branch 或目录关键字（默认 master）")
    parser.add_argument("--commit-id", help="指定 commit id（取前8位）")
    parser.add_argument("--download-only", action="store_true", help="只下载")
    parser.add_argument("--no-cleanup", action="store_true", help="保留旧包")
    parser.add_argument("--date", help="指定日期 YYYYMMDD")
    args = parser.parse_args()

    update_package(args.package_name,
                   args.whl_keyword,
                   args.branch,
                   download_only=args.download_only,
                   cleanup_old=not args.no_cleanup,
                   specified_day=args.date,
                   commit_id=args.commit_id)
