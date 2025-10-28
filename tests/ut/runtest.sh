#!/bin/sh
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
PROJECT_PATH=${BASE_PATH}/../../inferrt
if [ $BUILD_PATH ]; then
  echo "BUILD_PATH = $BUILD_PATH"
else
  BUILD_PATH=$(find ${PROJECT_PATH}/build/lib.* -maxdepth 0 -type d | head -n1)/mrt
  if [ -z "$BUILD_PATH" ]; then
    echo "Error: failed to locate build dir in ${PROJECT_PATH}/build/lib.*" >&2
    exit 1
  fi
  echo "BUILD_PATH = $BUILD_PATH"
fi
export LD_LIBRARY_PATH=${BUILD_PATH}/lib:$LD_LIBRARY_PATH
if [ "$TEST_TYPE" = "cpp" ]; then
    TEST_PATH=${BUILD_PATH}/bin/tests
    TEST_CASES=""
    for test_file in $(find "$TEST_PATH" -type f -name "run_*_test" -executable); do
        TEST_CASES="$TEST_CASES $test_file"
    done
else
    echo "Python UT testcases are not implemented yet, skipping."
    exit 0
fi

set +e

RET=0
echo "===================================="
for test_case in $TEST_CASES; do
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
exit $RET