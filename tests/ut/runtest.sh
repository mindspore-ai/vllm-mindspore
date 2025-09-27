#!/bin/bash
# Run UT testcases

set -e

if [ $# -ne 1 ] || { [ "$1" != "cpp" ] && [ "$1" != "python" ]; }; then
    echo "Error: Parameter must be 'cpp' or 'python'"
    echo "Usage: $0 {cpp|python}"
    exit 1
fi
TEST_TYPE=$1

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

if [ "$TEST_TYPE" = "cpp" ]; then
    TEST_PATH=${BUILD_PATH}/tests/ut/cpp
    TEST_CASES=(
        "./run_storage_cpu_test"
        "./run_hardware_cpu_test"
    )
else
    echo "Python UT testcases are not implemented yet, skipping."
    exit 0
fi

cd ${TEST_PATH}

set +e

RET=0
echo "===================================="
for test_case in "${TEST_CASES[@]}"; do
  echo "=== Running $test_case ==="
  $test_case
  status=$?
  if [ $status != 0 ]; then
    RET=$status
    exit $RET
  fi
done

echo "All testcases completed."
echo "===================================="
cd -
exit $RET
