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
import base64
import io
import os
import shlex
import site
import subprocess
import threading
import time
from datetime import datetime
from queue import Empty, Queue

import matplotlib.pyplot as plt
import pandas as pd
import regex as re
import requests
from acc import aisbench_test

COLOR_LIST = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# ===== 包和 commit 功能 =====
PACKAGES = [
    ("vllm-mindspore", "vllm_mindspore"),
    ("mindspore", "mindspore"),
    ("mindspore_gs", "mindspore_gs"),
    ("mindformers", "mindformers"),
]


def get_commit_info(
    app_name=None,
    pkg_dir_name=None,
    repo_path=None,
    env_prefix=None,
    short=True,
):
    commit_id = None
    branch = None
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
        sha_match = re.search(r"\b[0-9a-f]{7,40}\b", content)
        if sha_match:
            commit_id = sha_match.group(0)
            if short:
                commit_id = commit_id[:7]
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


def get_commit_info_for_packages(short=True):
    info = {}
    for app_name, pkg_dir in PACKAGES:
        commit = get_commit_info(app_name=app_name,
                                 pkg_dir_name=pkg_dir,
                                 short=short)
        info[app_name] = commit
    return info


# 允许字母数字、下划线、中划线、点、斜杠、冒号、@、{}、'、"、=、,、空格
ARGS_RE = re.compile(r'^[\w\-\./:@{}\'\"=, ]+$')


def validate_args(name: str):
    if not isinstance(name, str) or not ARGS_RE.match(name):
        raise ValueError("args not valid")


# ===== vLLM benchmark 功能 =====
def start_vllm_mindspore_server(model: str, serve_args: str):
    validate_args(model)
    validate_args(serve_args)
    cmd = f"vllm-mindspore serve {model} {serve_args} --port 8333"
    print(f"🚀 Starting vLLM-mindspore server: {cmd}")
    process = subprocess.Popen(shlex.split(cmd),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               text=True)
    url = "http://0.0.0.0:8333/v1/models"
    # 用于收集日志的队列
    log_queue = Queue()
    logs = ""

    def read_output():
        """在单独线程中读取输出"""
        while True:
            output = process.stdout.readline()
            if output:
                print(output, end='')  # 实时打印
                log_queue.put(output)
            elif process.poll() is not None:  # 进程结束
                break

    # 启动输出读取线程
    output_thread = threading.Thread(target=read_output, daemon=True)
    output_thread.start()

    for _ in range(1800):
        try:
            # 收集当前所有可用的日志
            while True:
                try:
                    log_line = log_queue.get_nowait()
                    logs += log_line
                except Empty:
                    break
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print(f"✅ vLLM-mindspore server for {model} is ready!")
                return process, logs
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"❌ vLLM-mindspore server for {model} failed to start.")


def run_benchmark_serving(model: str, bench_args: str):
    validate_args(model)
    validate_args(bench_args)
    cmd = (f"vllm-mindspore bench serve --base-url=http://0.0.0.0:8333 "
           f"--model={model} {bench_args}")
    print(f"▶️ Running benchmark: {cmd}")
    output = subprocess.check_output(shlex.split(cmd), text=True)
    return output


def parse_serving_output(output: str):
    metrics = {}
    mapping = {
        "Loading weights took": "load_weight_time(s)",
        "init engine": "init_engine_time(s)",
        "Available KV cache memory": "kv_cache_memory(GB)",
        "reserved for KV Cache is": "kv_cache_memory(GB)",
    }
    pattern = re.compile(
        r'(Loading weights took|Available KV cache memory|'
        r'reserved for KV Cache is|init engine).*?(\d+\.\d+)|'
        r'Model loading took.*?and (\d+\.\d+) seconds', re.IGNORECASE)
    matches = pattern.findall(output)
    for match in matches:
        # 处理前四种情况
        if match[0] and match[1]:
            key = match[0]
            if key in mapping:
                try:
                    metrics[mapping[key]] = float(match[1])
                except ValueError:
                    metrics[mapping[key]] = match[1]
        # 处理Model loading took情况
        elif match[2]:
            try:
                metrics["model_loading_time(s)"] = float(match[2])
            except ValueError:
                metrics["model_loading_time(s)"] = match[2]

    return metrics


def parse_bench_output(output: str):
    metrics = {}
    mapping = {
        "Request throughput (req/s)": "request_throughput(req/s)",
        "Output token throughput (tok/s)": "output_token_throughput(tok/s)",
        "Total Token throughput (tok/s)": "total_token_throughput(tok/s)",
        "Mean TTFT (ms)": "mean_ttft(ms)",
        "Mean TPOT (ms)": "mean_tpot(ms)",
        "Mean ITL (ms)": "mean_itl(ms)",
    }
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip().split()[0]
        if key in mapping:
            try:
                metrics[mapping[key]] = float(value)
            except ValueError:
                metrics[mapping[key]] = value

    return metrics


def sanitize_name(name: str) -> str:
    return (name.replace("/", "_").replace(":", "_").replace(" ", "_"))


def save_metrics(
    display_name: str,
    metrics: dict,
    serve_args: str,
    bench_args: str,
    base_dir="results",
):
    safe_name = sanitize_name(display_name)
    model_dir = os.path.join(base_dir, safe_name)
    os.makedirs(model_dir, exist_ok=True)
    csv_file = os.path.join(model_dir, "benchmark.csv")
    file_exists = os.path.isfile(csv_file)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp": now,
        "serve_args": serve_args,
        "bench_args": bench_args,
    }
    package_info = get_commit_info_for_packages()
    for pkg_name, commit in package_info.items():
        row[f"{pkg_name}"] = commit
    row.update(metrics)
    df_new = pd.DataFrame([row])
    if file_exists:
        df_old = pd.read_csv(csv_file)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(csv_file, index=False)
    print(f"✅ Saved metrics to {csv_file}")
    return csv_file, model_dir


# ===== 模型页面 HTML（响应式并列显示） =====
def generate_model_html(display_name: str, model_dir: str, csv_file: str):
    df = pd.read_csv(csv_file)
    # 按时间戳排序
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp', ascending=False)

    throughput_cols = [
        "request_throughput(req/s)",
        "output_token_throughput(tok/s)",
        "total_token_throughput(tok/s)",
    ]
    latency_cols = ["mean_ttft(ms)", "mean_tpot(ms)", "mean_itl(ms)"]

    def plot_and_encode(y_cols, ylabel):
        fig, ax = plt.subplots(figsize=(6, 3))
        for i, c in enumerate(y_cols):
            if c not in df:
                continue
            y = pd.to_numeric(df[c], errors="coerce").fillna(0)
            x = pd.to_datetime(df["timestamp"])
            ax.plot(
                x,
                y,
                label=c,
                color=COLOR_LIST[i % len(COLOR_LIST)],
            )
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    throughput_img = plot_and_encode(throughput_cols, "TPS")
    latency_img = plot_and_encode(latency_cols, "Latency (ms)")

    html = f"""
<html>
<head>
<meta charset="utf-8">
<title>Benchmark - {display_name}</title>
<link rel="stylesheet"
 href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css">
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script
 src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
<style>
body{{font-family:Arial,sans-serif; margin:20px;}}
.graph-container{{display:flex; flex-wrap:wrap; gap:4%;
 justify-content:space-between;}}
.graph-container div{{flex: 0 0 48%;}}
.graph-container img{{width:100%; height:auto;}}
h1{{color:#1f77b4;}}

/* 固定表头样式 */
.table-container {{
    max-height: calc(100vh - 100px);
    overflow: auto;
}}

#metrics thead th {{
    position: sticky;
    top: 0;
    background-color: #f8f9fa;
    z-index: 10;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}}
</style>
</head>
<body>
<h1>Benchmark - {display_name}</h1>

<div class="graph-container">
    <div>
        <h2>Throughput</h2>
        <img src="data:image/png;base64,{throughput_img}">
    </div>
    <div>
        <h2>Latency</h2>
        <img src="data:image/png;base64,{latency_img}">
    </div>
</div>

<h2>All Metrics</h2>
<div class="table-container">
<table id="metrics" class="display" style="width:100%">
    <thead>
        <tr>
            {"".join(
                f"<th>{c}</th>"
                for c in df.columns if c not in ['serve_args', 'bench_args']
            )}
        </tr>
    </thead>
    <tbody>
        {
            ''.join(
                "<tr>"
                + "".join(
                    f"<td>{row[c]}</td>"
                    for c in df.columns if c not in ['serve_args', 'bench_args']
                )
                + "</tr>"
                for _, row in df.iterrows()
            )
        }
    </tbody>
</table>
</div>

<script>
$(document).ready(function() {{
    $('#metrics').DataTable({{
        pageLength: 30,
        order: [], // 禁用初始排序，保持按时间戳倒序
        // 按时间戳列（第一列）倒序排序
        columnDefs: [
            {{ targets: 0, type: 'date' }} // 确保时间戳列按日期类型排序
        ]
    }});
}});
</script>
</body>
</html>
"""
    html_path = os.path.join(model_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Generated HTML report (matplotlib, side-by-side): {html_path}")
    return html_path


def parse_arguments(args_string):
    args_dict = {}
    # 以空格分割整个输入字符串
    parts = args_string.split()
    for part in parts:
        # 如果部分以'--'开头，则尝试提取key和value
        if part.startswith('--'):
            # 从第3个字符开始，分割key和value
            key_value = part[2:].split('=', 1)
            if len(key_value) == 2:
                key, value = key_value
                args_dict[key] = value
    return args_dict


# ===== 首页 Dashboard HTML =====
def generate_index_html(base_dir="results"):
    models = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    html_rows = []
    for m in models:
        csv_file = os.path.join(base_dir, m, "benchmark.csv")
        if not os.path.isfile(csv_file):
            continue
        df = pd.read_csv(csv_file)
        # 按时间戳排序
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)

        perf_columns = [
            c for c in df.columns if c not in [
                "timestamp",
                "serve_args",
                "bench_args",
                "vllm-mindspore",
                "mindspore",
                "mindspore_gs",
                "mindformers",
            ]
        ]
        df[perf_columns] = df[perf_columns].apply(pd.to_numeric,
                                                  errors="coerce")
        latest = df.iloc[0]
        display_name = m

        row_cells = (f"<td>{latest['timestamp']}</td>")
        bench_args_dict = parse_arguments(latest['bench_args'])
        row_cells += (f"<td>{bench_args_dict['num-prompts']}</td>"
                      f"<td>{bench_args_dict['random-input-len']}</td>"
                      f"<td>{bench_args_dict['random-output-len']}</td>")
        for c in perf_columns:
            fig, ax = plt.subplots(figsize=(1.5, 0.3))
            color = COLOR_LIST[hash(c) % len(COLOR_LIST)]
            y = df[c].fillna(0)
            x = range(len(y))
            ax.plot(x, y, color=color, linewidth=1)
            ax.axis("off")
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            row_cells += (
                f"<td>{latest[c]}<br>"
                f"<img src='data:image/png;base64,{img_base64}'></td>")
        # 包 commit id
        for pkg in [
                "vllm-mindspore",
                "mindspore",
                "mindspore_gs",
                "mindformers",
        ]:
            row_cells += f"<td>{latest.get(pkg,'')}</td>"
        row_cells += (f"<td>{latest['serve_args']}</td>"
                      f"<td>{latest['bench_args']}</td>")
        html_rows.append(
            f"<tr><td><a href='{m}/index.html'>{display_name}</a></td>"
            f"{row_cells}</tr>")

    header_cells = ("<th>Model</th><th>Timestamp</th><th>batch_size</th>"
                    "<th>input_len</th><th>output_len</th>")
    if models:
        sample_csv = os.path.join(base_dir, models[0], "benchmark.csv")
        df_sample = pd.read_csv(sample_csv)
        perf_columns = [
            c for c in df_sample.columns if c not in [
                "timestamp",
                "serve_args",
                "bench_args",
                "vllm-mindspore",
                "mindspore",
                "mindspore_gs",
                "mindformers",
            ]
        ]
        for c in perf_columns:
            header_cells += f"<th>{c}</th>"
    for pkg in ["vllm-mindspore", "mindspore", "mindspore_gs", "mindformers"]:
        header_cells += f"<th>{pkg}</th>"
    header_cells += "<th>Serve Args</th><th>Bench Args</th>"

    html = f"""
<html>
<head>
<meta charset="utf-8">
<title>Benchmark Dashboard</title>
<link rel="stylesheet"
 href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css">
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script
 src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
<style>
body{{font-family:Arial,sans-serif;}}
h1{{color:#1f77b4;}}
td{{vertical-align:top;}}

/* 固定表头样式 */
.table-container {{
    max-height: calc(100vh - 100px);
    overflow: auto;
}}

#dashboard thead th {{
    position: sticky;
    top: 0;
    background-color: #f8f9fa;
    z-index: 10;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}}
</style>
</head>
<body>
<h1>Benchmark Dashboard</h1>
<div class="table-container">
<table id="dashboard" class="display" style="width:100%">
<thead><tr>{header_cells}</tr></thead>
<tbody>{''.join(html_rows)}</tbody>
</table>
</div>
<script>
$(document).ready(function(){{
    $('#dashboard').DataTable({{
        "pageLength": 30,
        "order": [[1, "desc"]] // 按时间戳列（第二列）倒序排序
    }});
}});
</script>
</body>
</html>
"""
    html_path = os.path.join(base_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Generated Dashboard HTML: {html_path}")
    return html_path


def run_eval(model, do_ceval=True, do_gsm8k=True):
    ais_bench_path = os.environ["AIS_BENCH_PATH"]
    validate_args(ais_bench_path)
    ceval_acc = 0
    gsm8k_acc = 0
    if do_ceval:
        print(f"\n===== Running ceval: {model}  =====")
        ceval_acc = aisbench_test(ais_bench_path,
                                  "vllm_api_general_chat",
                                  "ceval_gen_0_shot_cot_chat_prompt",
                                  path=args.model,
                                  model=args.model,
                                  host_port=8333,
                                  max_out_len=4096,
                                  batch_size=128)

    if do_gsm8k:
        print(f"\n===== Running gsm8k: {model}  =====")
        gsm8k_acc = aisbench_test(ais_bench_path,
                                  "vllm_api_general_chat",
                                  "gsm8k_gen_0_shot_cot_chat_prompt",
                                  path=args.model,
                                  model=args.model,
                                  host_port=8333,
                                  max_out_len=4096,
                                  batch_size=128)
    print("ceval acc:", ceval_acc)
    print("gsm8k acc:", gsm8k_acc)
    return ceval_acc, gsm8k_acc


# ===== 主程序 =====
"""
usage:
    one bench args:
    python benchmark_to_dashboard.py --model=gpt2 \
        --serve-args="--trust-remote-code" \
        --bench-args="--num-prompts=10 --random-input-len=20 \
            --dataset-name=random --trust-remote-code" \
        --display-name gpt2

    multiple bench args:
    python benchmark_to_dashboard.py --model=gpt2 \
        --serve-args="--trust-remote-code" \
        --bench-args="--num-prompts=10 --random-input-len=20 \
            --dataset-name=random --trust-remote-code; \
            --num-prompts=20 --random-input-len=20 \
            --dataset-name=random --trust-remote-code" \
        --display-name "gpt2-10;gpt2-20"
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="模型名称")
    parser.add_argument("--serve-args",
                        default="",
                        help="vllm-mindspore serve 参数")
    parser.add_argument("--bench-args",
                        default="",
                        help="多组 benchmark 参数，用 ';' 分隔")
    parser.add_argument("--display-name", default=None, help="多组用 ';' 分隔")
    args = parser.parse_args()

    bench_args_list = [
        b.strip() for b in args.bench_args.split(";") if b.strip()
    ]
    display_names = ([d.strip() for d in args.display_name.split(";")]
                     if args.display_name else [args.model] *
                     len(bench_args_list))

    if len(display_names) != len(bench_args_list):
        raise ValueError("⚠️ display-name 数量必须与 bench-args 数量一致")

    print(f"\n===== Starting server for model: {args.model} =====")
    server, server_log = start_vllm_mindspore_server(args.model,
                                                     args.serve_args)
    metrics_server = parse_serving_output(server_log)
    print("metrics_server:", metrics_server)

    try:
        do_ceval, do_gsm8k = True, True
        ceval_acc, gsm8k_acc = 0, 0
        for bench_args, display_name in zip(bench_args_list, display_names):
            print(f"\n===== Running benchmark: {display_name} "
                  f"({bench_args}) =====")
            output = run_benchmark_serving(args.model, bench_args)
            print(output)

            metrics = parse_bench_output(output)

            if do_ceval or do_gsm8k:
                ceval_acc, gsm8k_acc = run_eval(args.model, do_ceval, do_gsm8k)
            do_ceval, do_gsm8k = False, False
            metrics['ceval'], metrics['gsm8k'] = ceval_acc, gsm8k_acc
            metrics.update(metrics_server)

            csv_file, model_dir = save_metrics(display_name, metrics,
                                               args.serve_args, bench_args)
            generate_model_html(display_name, model_dir, csv_file)
        generate_index_html()
    finally:
        server.terminate()
        server.wait()
