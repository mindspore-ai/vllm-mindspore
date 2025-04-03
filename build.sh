#!/bin/bash

# Make sure the build directory exists.
BUILD_DIR=./build
if [ -d "$BUILD_DIR" ]; then
    echo "$BUILD_DIR already exists."
else
    mkdir $BUILD_DIR
fi

if [ ! -d "$BUILD_DIR" ]; then
    echo "error: $BUILD_DIR NOT exists."
    return
fi

# Make da execution.
cd $BUILD_DIR
cmake ..
make clean
make
cd -

# Run test.
echo "=============================="
echo "Run test case:"
echo "./da ./sample/da_llm_sample.da"
echo "=============================="
./$BUILD_DIR/da ./sample/da_llm_sample.da