#!/bin/bash
# Run ST testcases

set -e

if [ $# -ne 1 ] || { [ "$1" != "cpu" ] && [ "$1" != "ascend" ]; }; then
    echo "Error: Parameter must be 'cpu' or 'ascend'"
    echo "Usage: $0 {cpu|ascend}"
    exit 1
fi

if [ "$1" = "cpu" ]; then
    TEST_PLAT='cpu_linux'
else
    TEST_PLAT='platform_ascend'
fi

BASE_PATH=$(
  cd "$(dirname "$0")"
  pwd
)
PROJECT_PATH=${BASE_PATH}/../..
if [ $BUILD_PATH ]; then
  echo "BUILD_PATH = $BUILD_PATH"
else
  BUILD_PATH=${PROJECT_PATH}/build
  echo "BUILD_PATH = $BUILD_PATH"
fi

TEST_CASES=$(find tests/st -name test_*.py -type f | xargs -r grep -l 'arg_mark' | xargs -r grep -l "$TEST_PLAT" | xargs -r grep -l 'level0')

if [ -n "$TEST_CASES" ]; then
    echo "$TEST_CASES" | xargs python -m pytest -v -m "level0 and $TEST_PLAT"
else
    echo "No matching testcases found."
    exit 0
fi
