#!/bin/bash
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

set -euo pipefail

readonly SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
readonly CONFIG_FILE="$SCRIPT_DIR/.jenkins/test/config/dependent_packages.yaml"
readonly MF_DIR="$SCRIPT_DIR/mindformers"

FORCE_REINSTALL=false

log() { echo "========= $*"; }

pip_install() {
    if [ "$FORCE_REINSTALL" = true ]; then
        uv pip install --system --no-cache-dir --force-reinstall --trusted-host repo.mindspore.cn --trusted-host mirrors.aliyun.com -i https://mirrors.aliyun.com/pypi/simple "$@"
    else
        uv pip install --system --no-cache-dir --trusted-host repo.mindspore.cn --trusted-host mirrors.aliyun.com -i https://mirrors.aliyun.com/pypi/simple "$@"
    fi
}

get_config() { python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['$1'])" 2>/dev/null; }

get_mindformers_commit() {
    local mindformers_commit
    
    if [ -d ".git" ] && git submodule status tests/mindformers >/dev/null 2>&1; then
        mindformers_commit=$(git submodule status tests/mindformers | awk '{print $1}' | sed 's/^-//')
    else
        if [ -z "${MINDFORMERS_COMMIT:-}" ]; then
            log "Error: Script requires MINDFORMERS_COMMIT environment variable when running independently"
            exit 1
        fi
        mindformers_commit="$MINDFORMERS_COMMIT"
    fi
    
    if [ -z "$mindformers_commit" ]; then
        log "Failed to get mindformers commit"
        exit 1
    fi
    
    echo "$mindformers_commit"
}

get_package_url() {
    local package="$1"
    local arch="${2:-any}"
    local base_url=$(grep -A 1 -w "${package}:" "$CONFIG_FILE" | tail -n 1 | xargs)
    base_url="${base_url}${arch}/"
    
    if [[ "$package" == "mindspore" ]]; then
        local python_v="cp$(python3 --version 2>&1 | grep -oP 'Python \K\d+\.\d+' | tr -d .)"
        local wheel_url=$(curl -k -s "$base_url" | sed -n 's/.*href="\([^"]*\.whl\)".*/\1/p' | grep -v sha256 | grep "${python_v}-${python_v}" | head -n 1)
    else
        local wheel_url=$(curl -k -s "$base_url" | sed -n 's/.*href="\([^"]*\.whl\)".*/\1/p' | grep -v sha256 | head -n 1)
    fi
    
    echo "${base_url}${wheel_url}"
}

install_mindformers() {
    log "Installing mindformers"
    local commit_id=$(get_mindformers_commit)
    
    if [ "$FORCE_REINSTALL" = true ] && [ -d "$MF_DIR" ]; then
        log "Force reinstall: removing existing mindformers directory"
        rm -rf "$MF_DIR"
    fi
    
    if [ ! -d "$MF_DIR" ]; then
        git clone https://gitee.com/mindspore/mindformers.git "$MF_DIR"
    fi
    cd "$MF_DIR"
    git checkout "$commit_id"
    cd "$SCRIPT_DIR"
    
    log "Using mindformers commit: $commit_id"
}

cleanup_torch() {
    log "Cleaning torch packages"
    pip uninstall torch torch-npu torchvision torchaudio -y || true
}

cleanup_cache() {
    log "Cleaning up package caches"
    uv cache clean
    pip cache purge || true
    rm -rf ~/.cache/pip
    rm -rf ~/.local/share/uv
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Install vLLM-MindSpore dependencies"
    echo ""
    echo "OPTIONS:"
    echo "  -f, -F, --force-reinstall    Force reinstall all packages (with --force-reinstall)"
    echo "  -h, --help                  Show this help message and exit"
    echo ""
    echo "Examples:"
    echo "  $0                      Normal installation"
    echo "  $0 -f                   Force reinstall all packages"
    echo "  $0 -F                   Force reinstall all packages"
    echo "  $0 --force-reinstall    Force reinstall all packages"
}

main() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|-F|--force-reinstall)
                export FORCE_REINSTALL=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Error: Unknown option '$1'"
                echo ""
                usage
                exit 1
                ;;
        esac
    done

    if [ "$FORCE_REINSTALL" = true ]; then
        log "WARNING: Force reinstall mode enabled - all packages will be reinstalled"
        log "This will remove existing mindformers directory and reinstall all dependencies"
    fi

    pip install --trusted-host mirrors.aliyun.com -i https://mirrors.aliyun.com/pypi/simple uv

    [ ! -f "$CONFIG_FILE" ] && { echo "Config file not found: $CONFIG_FILE"; exit 1; }
    
    log "Starting dependency installation"
    
    local vllm_url=$(get_package_url "vllm" "any")
    local mindspore_url=$(get_package_url "mindspore" "unified/aarch64")
    local msadapter_url=$(get_package_url "msadapter" "any")
    local mindspore_gs_url=$(get_package_url "mindspore_gs" "any")
    
    log "Package URLs:"
    log "vLLM: $vllm_url"
    log "MindSpore: $mindspore_url"
    log "msadapter: $msadapter_url"
    log "mindspore-gs: $mindspore_gs_url"
    
    pip_install "$vllm_url"
    cleanup_torch
    pip_install "$mindspore_url" "$msadapter_url" "$mindspore_gs_url"
    install_mindformers

    cleanup_cache
    
    log "All dependencies installed successfully!"
    log ""
    log "To use vLLM-MindSpore, environment variables need to be configured:"
    log "- export PYTHONPATH=\"$MF_DIR/:\$PYTHONPATH\""
    log "- export vLLM_MODEL_BACKEND=MindFormers"
    log ""
    
    if [ "${AUTO_BUILD:-}" = "1" ]; then
        echo "export PYTHONPATH=\"$MF_DIR/:\$PYTHONPATH\"" >> ~/.bashrc
        echo "export vLLM_MODEL_BACKEND=MindFormers" >> ~/.bashrc
        log "Environment variables added to ~/.bashrc (automated build)"
    else
        echo -n "Add these to ~/.bashrc automatically? (y/N): "
        read -r response
        
        if [[ "$response" == "y" || "$response" == "Y" ]]; then
            if grep -q "export PYTHONPATH.*mindformers" ~/.bashrc 2>/dev/null && grep -q "export vLLM_MODEL_BACKEND=MindFormers" ~/.bashrc 2>/dev/null; then
                log "Environment variables already exist in ~/.bashrc, skipping"
            else
                echo "export PYTHONPATH=\"$MF_DIR/:\$PYTHONPATH\"" >> ~/.bashrc
                echo "export vLLM_MODEL_BACKEND=MindFormers" >> ~/.bashrc
                log "Environment variables added to ~/.bashrc"
                log "Run: source ~/.bashrc"
            fi
        else
            log "Please add the environment variables manually and run 'source ~/.bashrc'"
        fi
    fi
}

main "$@"
