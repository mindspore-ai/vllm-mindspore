#!/bin/bash
set -e


CURRENT_PATH=$(pwd)
SCRIPT_PATH=$(dirname "$0")

DALANG_PATH=$CURRENT_PATH/dalang
DAPY_PATH=$CURRENT_PATH/dapy

BUILD_DIR=$CURRENT_PATH/build
BUILD_DIR_DALANG=$BUILD_DIR/dalang
BUILD_DIR_DAPY=$BUILD_DIR/dapy

# Make sure the build directory exists
make_sure_build_dir()
{
    if [ -d "$1" ]; then
        echo "$1 already exists."
    else
        mkdir -p $1
    fi

    if [ ! -d "$1" ]; then
        echo "error: $1 NOT exists."
        return
    fi
}
make_sure_build_dir $BUILD_DIR_DALANG
make_sure_build_dir $BUILD_DIR_DAPY


##################################################
# Step 1:
# Build DALANG shared library and execution
##################################################

# Make da execution and shared library
cd $BUILD_DIR_DALANG
cmake $DALANG_PATH
make

# Run test
echo "=============================="
echo "Run da execution test case:"
echo "./da sample/da_llm_sample.da"
echo "=============================="
$BUILD_DIR_DALANG/da $DALANG_PATH/sample/da_llm_sample.da


##################################################
# Step 2:
# Build DAPY shared library and execution
##################################################

# Set dalang shared library path for _dapy linking
DALANG_LIBRARIES="$BUILD_DIR_DALANG/libdalang.so"
echo "DALANG_LIBRARIES=$DALANG_LIBRARIES"

# Update pybind11 submodule
update_pybind11_submodule()
{
    if [ -d "pybind11" ]; then
        echo "pybind11 already exists."
        git submodule update --init
    else
        echo "pybind11 not found, start to clone."
        # Change github repo to gitee's: https://github.com/pybind/pybind11 ==> https://gitee.com/mirrors/pybind11
        git submodule add --force -b stable https://gitee.com/mirrors/pybind11 pybind11
        git submodule update --init
    fi
}
cd $DAPY_PATH
update_pybind11_submodule
PYBIND11_PATH=$DAPY_PATH/pybind11

# Make _dapy python module
cd $BUILD_DIR_DAPY
cmake $DAPY_PATH -DDALANG_LIBRARIES=$DALANG_LIBRARIES -DPYBIND11_PATH=$PYBIND11_PATH
make

# Run test
echo "=============================="
echo "Run dapy test case:"
echo "python check_api.py --dump=True"
echo "=============================="
export PYTHONPATH=$BUILD_DIR_DAPY:$DAPY_PATH/python
echo "PYTHONPATH=$PYTHONPATH"
python $DAPY_PATH/python/check_api.py --dump=True