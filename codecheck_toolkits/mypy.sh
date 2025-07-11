#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

CI=${1:-0}
PYTHON_VERSION=${2:-local}

if [ "$CI" -eq 1 ]; then
    set -e
fi

if [ $PYTHON_VERSION == "local" ]; then
    PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
fi

MERGEBASE="$(git merge-base origin/develop HEAD)"

if ! git diff --cached --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &> /dev/null; then
  git diff --cached --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs \
  mypy --follow-imports skip --python-version "${PYTHON_VERSION}" --config-file "codecheck_toolkits/pyproject.toml"
fi
