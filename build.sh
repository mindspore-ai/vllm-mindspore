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
BUILD_DIR=./build

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
CMAKE_DALANG_LIB_PATH="$(pwd)/$BUILD_DIR/libdalang.so"
echo "CMAKE_DALANG_LIB_PATH=$CMAKE_DALANG_LIB_PATH"


##################################################
# Build DAPY shared library and execution
##################################################
cd $CURRENT_PATH/python

update_pybind11_submodule()
{
    if [ -d "pybind11" ]; then
        echo "pybind11 already exists."
        git submodule update --init
    else
        echo "no pybind11 found, start to clone."
        git submodule add -b stable https://github.com/pybind/pybind11 pybind11
        git submodule update --init
    fi
}
update_pybind11_submodule

# Make da execution and shared library
make_sure_build_dir
cd $BUILD_DIR

# Set Python directories for CMake
CMAKE_PYTHON_INCLUDE_DIR=$(python -m pybind11 --includes)
echo "CMAKE_PYTHON_INCLUDE_DIR=$CMAKE_PYTHON_INCLUDE_DIR"
CMAKE_PYTHON_CONFIG_CFLAGS=$(python3-config --cflags)
echo "CMAKE_PYTHON_CONFIG_CFLAGS=$CMAKE_PYTHON_CONFIG_CFLAGS"
CMAKE_PYTHON_CONFIG_LDFLAGS=$(python3-config --ldflags)
echo "CMAKE_PYTHON_CONFIG_LDFLAGS=$CMAKE_PYTHON_CONFIG_LDFLAGS"

cmake ..\
  -DCMAKE_DALANG_LIB_PATH=$CMAKE_DALANG_LIB_PATH\
  -DCMAKE_PYTHON_INCLUDE_DIR=$CMAKE_PYTHON_INCLUDE_DIR\
  -DCMAKE_PYTHON_CONFIG_LDFLAGS=$CMAKE_PYTHON_CONFIG_LDFLAGS
make
cd $CURRENT_PATH
