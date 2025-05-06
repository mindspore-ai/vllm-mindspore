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

DART_PATH=$CURRENT_PATH
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


##################################################
# Make da & dapy execution and shared library
##################################################
cd $BUILD_DIR
if [[ $INC_BUILD != 1 ]]; then
    rm $BUILD_DIR/* -rf
    cmake $DART_PATH $DALANG_CMAKE_ARGS $DAPY_CMAKE_ARGS
fi
make


##################################################
# Run essential test
##################################################
# Run dalang test
echo "=============================="
echo "Run da execution test cases:"
echo "# 1/2: ./da sample/fibonacci_20.da"
$BUILD_DIR/dalang/da $DART_PATH/dalang/sample/fibonacci_20.da
echo "# 2/2: ./da sample/da_llm_sample.da"
$BUILD_DIR/dalang/da $DART_PATH/dalang/sample/da_llm_sample.da
echo "=============================="

# Run dapy test
echo "=============================="
echo "Run dapy test case:"
echo "python check_api.py"
echo "=============================="
export PYTHONPATH=$BUILD_DIR/dapy:$DART_PATH/dapy/python
echo "PYTHONPATH=$PYTHONPATH"
python $DART_PATH/dapy/python/check_api.py