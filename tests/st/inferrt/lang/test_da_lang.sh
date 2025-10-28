#!/bin/bash
set -e

##################################################
# Run da lang test
##################################################
echo "=============================="
echo "Run da execution test cases:"
echo "# 1/2: ./da sample/fibonacci_20.da"
$BUILD_DIR/inferrt/src/da $INFERRT_PATH/inferrt/src/lang/sample/fibonacci_20.da
echo "# 2/2: ./da sample/da_llm_sample.da"
$BUILD_DIR/inferrt/src/da $INFERRT_PATH/inferrt/src/lang/sample/da_llm_sample.da
echo "=============================="
