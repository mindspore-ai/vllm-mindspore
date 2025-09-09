#!/bin/bash
set -e


##################################################
# Process build options
##################################################
usage()
{
  echo "Usage:"
  echo "bash build.sh [-D] [-d [lexer,parser,compiler,vm,tensor,ops,pass,runtime,py]] [-i] [-h]"
  echo ""
  echo "Options:"
  echo "    -d Enable log print of modules, separated by comma(eg. -d parser,compiler),"
  echo "       default off"
  echo "    -h Print usage"
  echo "    -i Enable increment building, default off"
  echo "    -D Debug version, default release version"
  echo "    -a Enable pytorch aten kernel, default off"
  echo "    -t Build and run tests, default off"
  echo "    -e Enable Ascend backend, default off"
}

process_options()
{
    while getopts 'Dd:hiate' OPT; do
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
            t) export BUILD_TESTS=1;;
            e) export ENABLE_ASCEND=1;;
            a)
                export ENABLE_KERNEL_ATEN="-DENABLE_KERNEL_ATEN=on"
                export TEST_TORCH=1
                ;;
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
INFERRT_CMAKE_ARGS="${INFERRT_CMAKE_ARGS} $DEBUG $DEBUG_LOG_OUT $ENABLE_KERNEL_ATEN"
DAPY_CMAKE_ARGS="${DAPY_CMAKE_ARGS} $DEBUG $DEBUG_LOG_OUT $ENABLE_KERNEL_ATEN"

if [[ $BUILD_TESTS == 1 ]]; then
    INFERRT_CMAKE_ARGS="${INFERRT_CMAKE_ARGS} -DBUILD_TESTS=on"
fi

if [[ $ENABLE_ASCEND == 1 ]]; then
    INFERRT_CMAKE_ARGS="${INFERRT_CMAKE_ARGS} -DENABLE_ASCEND=on"
fi

##################################################
# Prepare source and build directories
##################################################
CURRENT_PATH=$(pwd)
SCRIPT_PATH=$(dirname "$0")

INFERRT_PATH=$CURRENT_PATH
BUILD_DIR=$CURRENT_PATH/build

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
make_sure_build_dir $BUILD_DIR

# Try using ccache
if type -P ccache &>/dev/null; then
    echo "ccache found, using ccache for building."
    export CCACHE_CMAKE_ARGS="-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
else
    echo "ccache not found, using default compiler."
    export CCACHE_CMAKE_ARGS=""
fi

##################################################
# Make da & dapy execution and shared library
##################################################
cd $BUILD_DIR
if [[ $INC_BUILD != 1 ]]; then
    rm $BUILD_DIR/* -rf
    cmake $INFERRT_PATH $CCACHE_CMAKE_ARGS $INFERRT_CMAKE_ARGS $DAPY_CMAKE_ARGS
fi
make


##################################################
# Run essential test
##################################################
# Run inferrt test
export DART_KERNEL_LIB_PATH=$BUILD_DIR/inferrt/src/ops/dummy/libkernel_dummy.so
export DART_KERNEL_LIB_NAME=Dummy
export DUMMY_RUN="on"
echo "=============================="
echo "Run da execution test cases:"
echo "# 1/2: ./da sample/fibonacci_20.da"
$BUILD_DIR/inferrt/src/da $INFERRT_PATH/inferrt/src/lang/sample/fibonacci_20.da
echo "# 2/2: ./da sample/da_llm_sample.da"
$BUILD_DIR/inferrt/src/da $INFERRT_PATH/inferrt/src/lang/sample/da_llm_sample.da
echo "=============================="

# Run hardware test
if [[ $BUILD_TESTS == 1 ]]; then
    echo "=============================="
    echo "Run test case:"
    if [[ $ENABLE_ASCEND == 1 ]]; then
        echo "Ascend backend test case"
        ./tests/hardware_ascend_test_obj
    else
        echo "CPU backend test case"
    fi
    echo "Tests completed."
    echo "=============================="
fi

# Run python test
echo "=============================="
echo "Run python test case:"
echo "python check_api.py"
echo "=============================="
export PYTHONPATH=$BUILD_DIR/inferrt/src/pybind/mrt:$INFERRT_PATH/inferrt/python
echo "PYTHONPATH=$PYTHONPATH"
python $INFERRT_PATH/inferrt/python/check_api.py

# Run pytorch backend test
if [[ $TEST_TORCH == 1 ]]; then
    echo "=============================="
    echo "Run pytorch backend test case:"
    echo "python check_backend.py"
    echo "=============================="
    export PYTHONPATH=$BUILD_DIR/inferrt/src/pybind/mrt:$BUILD_DIR/inferrt/src/pybind/mrt_torch:$INFERRT_PATH/inferrt/python
    export DART_KERNEL_LIB_PATH=$BUILD_DIR/inferrt/src/ops/cpu/aten/libkernel_aten.so
    export DART_KERNEL_LIB_NAME=Aten
    python $INFERRT_PATH/inferrt/python/check_backend.py
fi
