#!/bin/bash

##################################################
# Build DALANG shared library and execution
##################################################
# Make sure the build directory exists.
CURRENT_PATH=$(pwd)
SCRIPT_PATH=$(dirname "$0")
echo "CURRENT_PATH=$CURRENT_PATH"
echo "SCRIPTS_PATH=$SCRIPTS_PATH"

DALANG_PATH=$CURRENT_PATH/dalang
cd $DALANG_PATH
BUILD_DIR=build

make_sure_build_dir()
{
    if [ -d "$BUILD_DIR" ]; then
        echo "$BUILD_DIR already exists."
    else
        mkdir $BUILD_DIR
    fi

    if [ ! -d "$BUILD_DIR" ]; then
        echo "error: $BUILD_DIR NOT exists."
        return
    fi
}

# Make da execution and shared library
make_sure_build_dir
cd $BUILD_DIR
cmake ..
make
cd $DALANG_PATH

# Run test.
echo "=============================="
echo "Run test case:"
echo "./da ./sample/da_llm_sample.da"
echo "=============================="
./$BUILD_DIR/da ./sample/da_llm_sample.da

# Set dalang shared library path
DALANG_LIBRARIES="$(pwd)/$BUILD_DIR/libdalang.so"
echo "DALANG_LIBRARIES=$DALANG_LIBRARIES"


##################################################
# Build DAPY shared library and execution
##################################################
cd $CURRENT_PATH/dapy

update_pybind11_submodule()
{
    if [ -d "pybind11" ]; then
        echo "pybind11 already exists."
        git submodule update --init
    else
        echo "no pybind11 found, start to clone."
        # Change github repo to gitee's: https://github.com/pybind/pybind11 ==> https://gitee.com/mirrors/pybind11
        git submodule add --force -b stable https://gitee.com/mirrors/pybind11 pybind11
        git submodule update --init
    fi
}
update_pybind11_submodule

# Make da execution and shared library
make_sure_build_dir
cd $BUILD_DIR

# Check Python directories for CMake
PYTHON_INCLUDE_DIR=$(python3 -m pybind11 --includes)
echo "PYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR"
PYTHON_CONFIG_CFLAGS=$(python3-config --cflags)
echo "PYTHON_CONFIG_CFLAGS=$PYTHON_CONFIG_CFLAGS"
PYTHON_CONFIG_LDFLAGS=$(python3-config --ldflags)
echo "PYTHON_CONFIG_LDFLAGS=$PYTHON_CONFIG_LDFLAGS"
DALANG_PY_PACKAGE_NAME=_dapy$(python3-config --extension-suffix)
echo "DALANG_PY_PACKAGE_NAME=$DALANG_PY_PACKAGE_NAME"

cmake .. -DDALANG_LIBRARIES=$DALANG_LIBRARIES
make
cd $CURRENT_PATH
