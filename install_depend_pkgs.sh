#!/bin/bash

set -euo pipefail

readonly SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
readonly CONFIG_FILE="$SCRIPT_DIR/.jenkins/test/config/dependent_packages.yaml"
readonly WORK_DIR="/workspace/install_depend_pkgs"
readonly MF_DIR="/workspace/mindformers"

log() { echo "========= $*"; }

log "Installing uv package manager"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

alias pip="uv pip"
get_config() { python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['$1'])" 2>/dev/null; }

get_package_url() {
    local package="$1"
    local arch="${2:-any}"
    local base_url=$(grep -A 1 -w "${package}:" "$CONFIG_FILE" | tail -n 1 | xargs)
    base_url="${base_url}${arch}/"
    
    if [[ "$package" == "mindspore" && "$arch" == "unified/aarch64" ]]; then
        local wheel_url=$(curl -s "$base_url" | sed -n 's/.*href="\([^"]*\.whl\)".*/\1/p' | grep -v sha256 | grep "cp311-cp311" | head -n 1)
    else
        local wheel_url=$(curl -s "$base_url" | sed -n 's/.*href="\([^"]*\.whl\)".*/\1/p' | grep -v sha256 | head -n 1)
    fi
    
    echo "${base_url}${wheel_url}"
}

install_from_url() {
    local name="$1" url="$2"
    log "Installing $name from $url"
    pip install --no-cache-dir "$url"
}

install_mindformers() {
    log "Installing mindformers"
    git clone https://gitee.com/mindspore/mindformers.git -b dev "$MF_DIR"
    cd "$MF_DIR"
    git fetch origin
    git checkout dev && git pull origin dev
    git checkout dfb8aa3a59401495b2d8c8c107d46fe0d36c949a
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

main() {
    [ ! -f "$CONFIG_FILE" ] && { echo "Config file not found: $CONFIG_FILE"; exit 1; }
    
    mkdir -p "$WORK_DIR"
    
    log "Starting dependency installation"
    
    local vllm_url=$(get_package_url "vllm" "any")
    local mindspore_url=$(get_package_url "mindspore" "unified/aarch64")
    local msadapter_url=$(get_package_url "msadapter" "any")
    local mindspore_gs_url=$(get_package_url "mindspore_gs" "any")
    
    log "Package URLs:"
    log "vLLM: $vllm_url"
    log "MindSpore: $mindspore_url"
    log "MSAdapter: $msadapter_url"
    log "MindSpore GS: $mindspore_gs_url"
    
    install_from_url "vllm" "$vllm_url"
    install_from_url "mindspore" "$mindspore_url"
    cleanup_torch
    install_from_url "msadapter" "$msadapter_url"
    install_mindformers
    install_from_url "mindspore-gs" "$mindspore_gs_url"
    
    cleanup_cache
    
    export PYTHONPATH="$MF_DIR/:$PYTHONPATH"
    export vLLM_MODEL_BACKEND=MindFormers
    
    log "All dependencies installed successfully!"
    log "PYTHONPATH set to: $PYTHONPATH"
    log "vLLM_MODEL_BACKEND set to: $vLLM_MODEL_BACKEND"
}

main "$@"