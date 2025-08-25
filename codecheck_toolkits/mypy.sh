#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

PYTHON_VERSION=${1:-local}

run_mypy() {
    if [ -z "$1" ]; then
      mypy --python-version "${PYTHON_VERSION}" "$@"
      return
    fi
    mypy --follow-imports skip --python-version "${PYTHON_VERSION}" "$@"
}

run_mypy
run_mypy tests/st
run_mypy vllm_mindspore/attention
run_mypy vllm_mindspore/distributed
run_mypy vllm_mindspore/engine
run_mypy vllm_mindspore/executor
run_mypy vllm_mindspore/inputs
run_mypy vllm_mindspore/lora
run_mypy vllm_mindspore/model_executor
run_mypy vllm_mindspore/worker
run_mypy vllm_mindspore/v1
