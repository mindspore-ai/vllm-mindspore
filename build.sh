#!/bin/bash
set -e


##################################################
# Process build options
##################################################
usage()
{
  echo "Usage:"
  echo "bash build.sh [-D] [-d [lexer,parser,compiler,vm,ir,rt,dapy]] [-i] [-h]"
  echo ""
  echo "Options:"
  echo "    -d Enable log print of modules, separated by comma(eg. -d parser,compiler),"
  echo "       default off"
  echo "    -h Print usage"
  echo "    -i Enable increment building, default off"
  echo "    -D Debug version, default release version"
}

process_options()
{
    while getopts 'Dd:hi' OPT; do
        case $OPT in
            D)
                # Debug version or not.
                # -D
                export DEBUG="-DDEBUG=on";;
            d)
                # Enable log out for modules.
                # -d lexer,parser,compiler,vm,ir,rt,dapy
                OPTARGS=(${OPTARG//,/ })
                for ARG in ${OPTARGS[@]}
                do
                    export DEBUG_LOG_OUT="$DEBUG_LOG_OUT -DDEBUG_LOG_OUT_$ARG=on"
                done
                ;;
            i) export INC_BUILD=1;;
            h)
                usage
                exit 0
                ;;
            ?)
                usage
                exit 1
                ;;
        esac
    done
}
process_options $@
DALANG_CMAKE_ARGS="${DALANG_CMAKE_ARGS} $DEBUG $DEBUG_LOG_OUT"
DAPY_CMAKE_ARGS="${DAPY_CMAKE_ARGS} $DEBUG $DEBUG_LOG_OUT"


##################################################
# Prepare source and build directories
##################################################
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
if [[ $INC_BUILD != 1 ]]; then
    rm $BUILD_DIR_DALANG/* -rf
    echo "DALANG_CMAKE_ARGS: $DALANG_CMAKE_ARGS"
    cmake $DALANG_PATH $DALANG_CMAKE_ARGS
fi
make

# Run test
echo "=============================="
echo "Run da execution test cases:"
echo "# 1/2: ./da sample/fibonacci_20.da"
$BUILD_DIR_DALANG/da $DALANG_PATH/sample/fibonacci_20.da
echo "# 2/2: ./da sample/da_llm_sample.da"
$BUILD_DIR_DALANG/da $DALANG_PATH/sample/da_llm_sample.da
echo "=============================="

##################################################
# Step 2:
# Build DAPY shared library and execution
##################################################

# Set dalang shared library path for _dapy linking
DALANG_LIBRARIES="$BUILD_DIR_DALANG/libdalang.so"
echo "DALANG_LIBRARIES=$DALANG_LIBRARIES"
DAPY_CMAKE_ARGS="${DAPY_CMAKE_ARGS} -DDALANG_LIBRARIES=$DALANG_LIBRARIES"

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
DAPY_CMAKE_ARGS="${DAPY_CMAKE_ARGS} -DPYBIND11_PATH=$PYBIND11_PATH"

# Make _dapy python module
cd $BUILD_DIR_DAPY
if [[ $INC_BUILD != 1 ]]; then
    rm $BUILD_DIR_DAPY/* -rf
    echo "DALANG_CMAKE_ARGS: $DAPY_CMAKE_ARGS"
    cmake $DAPY_PATH $DAPY_CMAKE_ARGS
fi
make

# Run test
echo "=============================="
echo "Run dapy test case:"
echo "python check_api.py"
echo "=============================="
export PYTHONPATH=$BUILD_DIR_DAPY:$DAPY_PATH/python
echo "PYTHONPATH=$PYTHONPATH"
python $DAPY_PATH/python/check_api.py