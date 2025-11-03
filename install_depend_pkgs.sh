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
readonly RELEASE_CONFIG_FILE="$SCRIPT_DIR/release_packages.yaml"
readonly MF_DIR="$SCRIPT_DIR/mindformers"

readonly PIP_TRUSTED_HOSTS="--trusted-host repo.mindspore.cn --trusted-host mirrors.aliyun.com --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com"
readonly PIP_INDEX="-i https://mirrors.aliyun.com/pypi/simple"

export UV_HTTP_TIMEOUT=600

FORCE_REINSTALL=false

# Detect architecture
ARCH=$([ "$(uname -m)" = "x86_64" ] && echo "x86_64" || echo "aarch64")

log() {
    local width=80
    local text="$*"
    
    if [ -z "$text" ]; then
        printf "\033[92m"
        printf "%0.s=" $(seq 1 $width)
        printf "\033[0m"
        echo
    else
        local padding=$(( (width - ${#text} - 2) / 2 ))
        printf "\033[92m"
        printf "%0.s=" $(seq 1 $padding)
        printf " %s " "$text"
        printf "%0.s=" $(seq 1 $((width - padding - ${#text} - 2)))
        printf "\033[0m"
        echo
    fi
}

log_package_url() {
    local package_name="$1"
    local url="$2"
    echo -e "\033[38;5;214m${package_name}:\033[0m"
    echo -e "  \033[38;5;214m${url}\033[0m"
}

pip_install() {
    log "Installing $*"
    if [ "$FORCE_REINSTALL" = true ]; then
        uv pip install --system --no-cache-dir --force-reinstall $PIP_TRUSTED_HOSTS $PIP_INDEX "$@"
    else
        uv pip install --system --no-cache-dir $PIP_TRUSTED_HOSTS $PIP_INDEX "$@"
    fi
}

get_config() { python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['$1'])" 2>/dev/null; }

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

get_obs_package_url() {
    local package="$1"
    local version="$2"
    local arch="${3:-any}"
    local base_url=$(grep -A 1 -w "${package}:" "$RELEASE_CONFIG_FILE" | tail -n 1 | xargs)

    if [ -z "${base_url:-}" ]; then
        local wheel_url=""
    else
        if [[ "$package" == "mindspore" ]]; then
            local python_v="cp$(python3 --version 2>&1 | grep -oP 'Python \K\d+\.\d+' | tr -d .)"
            local wheel_url="${base_url}/unified/${arch}/mindspore-${version}-${python_v}-${python_v}-linux_${arch}.whl"
        elif [[ "$package" == "mindformers" ]]; then
            local wheel_url="${base_url}/${arch}/mindformers-${version}-py3-none-${arch}.whl"
        elif [[ "$package" == "msadapter" ]]; then
            local wheel_url="${base_url}/${arch}/msadapter-${version}-py3-none-${arch}.whl"
        elif [[ "$package" == "vllm" ]]; then
            local wheel_url="${base_url}/${arch}/vllm-${version}.empty-py3-none-${arch}.whl"
        else
            local wheel_url=$(curl -k -s "$base_url" | sed -n 's/.*href="\([^"]*\.whl\)".*/\1/p' | grep -v sha256 | head -n 1)
        fi
    fi
    
    echo "${wheel_url}"
}

cleanup_package() {
    log "Cleaning $*"
    pip uninstall "$@" -y || true
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

    git config --global http.sslVerify false

    if [ "$FORCE_REINSTALL" = true ]; then
        log "WARNING: Force reinstall mode enabled - all packages will be reinstalled"
        log "This will remove existing mindformers directory and reinstall all dependencies"
    fi

    command -v uv &> /dev/null || pip install $PIP_TRUSTED_HOSTS $PIP_INDEX uv

    [ ! -f "$RELEASE_CONFIG_FILE" ] && { echo "Config file not found: $RELEASE_CONFIG_FILE"; exit 1; }
    
    log "Starting dependency installation"
    
    local vllm_url=$(get_package_url "vllm" "any")
    local mindspore_url=$(get_obs_package_url "mindspore" "2.7.1" "${ARCH}")
    local msadapter_url=$(get_package_url "msadapter" "any")
    local mindformers_url=$(get_obs_package_url "mindformers" "1.7.0" "any")
    
    if [ -z "${vllm_url:-}" ]; then
        local vllm_url=$(get_package_url "vllm" "any")
    fi
    if [ -z "${mindspore_url:-}" ]; then
        local mindspore_url=$(get_package_url "mindspore" "unified/${ARCH}")
    fi
    if [ -z "${msadapter_url:-}" ]; then
        local msadapter_url=$(get_package_url "msadapter" "any")
    fi
    if [ -z "${mindformers_url:-}" ]; then
        local mindformers_url=$(get_package_url "mindformers" "any")
    fi

    log "Package URLs:"
    log_package_url "vLLM" "$vllm_url"
    log_package_url "MindSpore" "$mindspore_url"
    log_package_url "mindformers" "$mindformers_url"
    log_package_url "msadapter" "$msadapter_url"

    # WARNING: do not adjust sequence of installation steps
    cleanup_package msadapter
    cleanup_package vllm
    pip_install "$vllm_url"
    pip_install "$mindspore_url"
    pip_install "$mindformers_url"
    pip_install "$msadapter_url"

    cleanup_cache

    git config --global http.sslVerify true
    
    log "All dependencies installed successfully!"
    log ""
}
main "$@"
