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
TEST_PATH=${PROJECT_PATH}/tests/st
echo "=== Collected Test Cases ==="
mapfile -t TEST_CASES < <(python -m pytest "$TEST_PATH" -m "level0 and $TEST_PLAT" --collect-only -q | grep -E 'test_.*\.py::' | tee /dev/tty)

if [ ${#TEST_CASES[@]} -eq 0 ]; then
    echo "No matching testcases found."
    exit 0
fi

for test_case in "${TEST_CASES[@]}"; do
  python -m pytest -s -v "$test_case"
done
echo "All testcases completed."