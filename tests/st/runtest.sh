#!/bin/bash
# Run ST testcases

set +e

if [ $# -gt 1 ] || { [ $# -eq 1 ] && [ "$1" != "cpu" ] && [ "$1" != "ascend" ]; }; then
    echo "Error: Invalid argument '$1'"
    echo "Usage: $0 [cpu|ascend]"
    echo "Run tests for specified platform, or both if no argument"
    exit 1
fi

BASE_PATH=$(
  cd "$(dirname "$0")"
  pwd
)
PROJECT_PATH=${BASE_PATH}/../..
TEST_PATH=${PROJECT_PATH}/tests/st

GREEN=$(tput setaf 2)
RESET=$(tput sgr0)

run_tests() {
    local platform=$1
    
    if [ "$platform" = "cpu" ]; then
        TEST_PLAT='cpu_linux'
    else
        TEST_PLAT='platform_ascend'
    fi

    echo "${GREEN}=============== Running $TEST_PLAT Tests ===============${RESET}"
    echo "${GREEN}=============== Finding Test Files ===============${RESET}"
    mapfile -t TEST_FILES < <(find "$TEST_PATH" -name "test_*.py" -type f | xargs -r grep -l 'arg_mark' | xargs -r grep -l "$TEST_PLAT" | xargs -r grep -l 'level0' | tee /dev/tty)
    echo "${GREEN}=============== Found ${#TEST_FILES[@]} Test Files ===============${RESET}"

    echo "${GREEN}=============== Collecting Test Cases ===============${RESET}"
    COLLECT_OUTPUT=$(python -m pytest "${TEST_FILES[@]}" -m "level0 and $TEST_PLAT" --collect-only -q 2>&1)
    EXIT_CODE=$?
    if [ $EXIT_CODE != 0 ]; then
        echo "$COLLECT_OUTPUT"
        exit $EXIT_CODE
    fi
    mapfile -t TEST_CASES < <(echo "$COLLECT_OUTPUT" | grep -E 'test_.*\.py::' | sed 's/\[.*//' | sort -u | tee /dev/tty)

    TOTAL_COUNT=${#TEST_CASES[@]}
    FAILED_COUNT=0
    RESULTS=()
    echo "${GREEN}=============== Collected ${TOTAL_COUNT} Test Cases ===============${RESET}"

    for test_case in "${TEST_CASES[@]}"; do
        python -m pytest -v "$test_case"
        status=$?

        # Distributed test cases require cleanup for resource release
        if [[ "$test_case" == *"distributed"* ]]; then
            cleanup_distributed_processes "$test_case"
        fi

        if [ $status -eq 0 ]; then
            result="PASSED"
        else
            result="FAILED"
            ((FAILED_COUNT++))
        fi
        RESULTS+=("$test_case|$TEST_PLAT|$result")
    done

    {
    printf "%s|%s|%s\n" "Testcase" "EnvType" "Result"
    for result_entry in "${RESULTS[@]}"; do
        echo "$result_entry"
    done
    } | column -t -s "|"

    echo "${GREEN}=============== Case Result Info ===============${RESET}"
    PASS_COUNT=$((TOTAL_COUNT - FAILED_COUNT))
    echo "Total Tests: $TOTAL_COUNT"
    echo "Total Failures: $FAILED_COUNT"
    echo "Total Success: $PASS_COUNT"

    return $FAILED_COUNT
}

cleanup_distributed_processes() {
    local test_case=$1
    sleep 5
    echo "Cleaning up residual processes for: $test_case"

    local op_name=$(echo "$test_case" | sed -n 's/.*test_check_\(.*\)_op.*/\1/p')
    pids=$(ps -ef | grep -E "test_check_${op_name}_op|test_${op_name}" | grep -v grep | awk '{print $2}')
    if [ -n "$pids" ]; then
        kill -9 $pids
    fi
}

if [ $# -eq 0 ]; then
    echo "${GREEN}=============== No parameter provided, running both ascend and cpu tests ===============${RESET}"
    
    echo "${GREEN}=============== First running ascend tests ===============${RESET}"
    run_tests "ascend"
    ASCEND_EXIT=$?

    echo "${GREEN}=============== Then running cpu tests ===============${RESET}"
    run_tests "cpu"
    CPU_EXIT=$?
    
    if [ $ASCEND_EXIT -ne 0 ] || [ $CPU_EXIT -ne 0 ]; then
        exit 1
    else
        exit 0
    fi
else
    run_tests "$1"
    exit $?
fi
