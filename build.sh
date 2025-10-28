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
  echo "    -t Build and run tests, default off"
  echo "    -f Enable frontend, default compile all frontend"
  echo "    -b Enable backend, default compile cpu backend"
  echo "    -e Enable download cmake compile dependency from gitee, default off"
  echo "    -O Enable optimizer, default off"
  echo "    -j Set the number of parallel build jobs, default 8"
}

process_options()
{
    # Default to CPU backend
    export ENABLE_CPU=1

    # Default compile all frontend
    export ENABLE_MINDSPORE_FRONT=1
    export ENABLE_TORCH_FRONT=1
    export BUILD_OPT=0 # Default disable optimizer for now
    export BUILD_JOBS=8 

    while getopts 'Dd:hitf:b:eOj:' OPT; do
        case $OPT in
            D)
                # Debug version or not.
                # -D
                export DEBUG="-DDEBUG=on";;
            d)
                # Enable log out for modules.
                # -d lexer,parser,compiler,vm,ir,rt,dapy
                IFS=',' read -ra OPTARGS <<< "$OPTARG"
                for ARG in "${OPTARGS[@]}"
                do
                    export DEBUG_LOG_OUT="$DEBUG_LOG_OUT -DDEBUG_LOG_OUT_$ARG=on"
                done
                ;;
            i) export INC_BUILD=1;;
            t) export BUILD_TESTS=1;;
            O) export BUILD_OPT=1;;
            j) export BUILD_JOBS=$OPTARG;;
            h)
                usage
                exit 0
                ;;
            b)
                if [ "$OPTARG" = "ascend" ]; then
                    export ENABLE_ASCEND=1
                elif [ "$OPTARG" = "cpu" ]; then
                    export ENABLE_CPU=1
                    unset ENABLE_ASCEND
                else
                    echo "Error: Invalid backend '$OPTARG'. Use 'ascend' or 'cpu'."
                    exit 1
                fi
                ;;
            f)
                unset ENABLE_MINDSPORE_FRONT
                unset ENABLE_TORCH_FRONT
                if [ "$OPTARG" = "ms" ]; then
                    export ENABLE_MINDSPORE_FRONT=1
                elif [ "$OPTARG" = "pt" ]; then
                    export ENABLE_TORCH_FRONT=1
                else
                    echo "Error: Invalid frontend '$OPTARG'. Use 'ms' or 'pt'."
                    exit 1
                fi
                ;;
            e) export ENABLE_GITEE=1;;
            ?)
                usage
                exit 1
                ;;
        esac
    done
}

process_options "$@"

##################################################
# Build third_party
##################################################
if [[ $INC_BUILD != 1 && $BUILD_OPT == 1 ]]; then
    BUILD_DIR=$(pwd)/build
    export LLVM_INSTALL_PREFIX="$BUILD_DIR/third_party/install/llvm"
    export TORCHMLIR_INSTALL_PREFIX="$BUILD_DIR/third_party/install/torch_mlir"
    bash "scripts/build_llvm.sh"
fi

# Install build dependencies
pip install -r requirements-build.txt

##################################################
# Build mopt
##################################################
if [[ $BUILD_OPT == 1 ]]; then
    pushd mopt

    # Clean previous builds
    if [[ $INC_BUILD != 1 ]]; then
        rm -rf build dist
    fi

    # Build the wheel
    python -m build --wheel --no-isolation

    popd
fi

##################################################
# Build mrt
##################################################
pushd inferrt

# Clean previous builds
if [[ $INC_BUILD != 1 ]]; then
    rm -rf build dist
fi

# Build the wheel
python -m build --wheel --no-isolation

popd

##################################################
# Gather generated wheel packages
##################################################
mkdir -p output
cp */dist/*.whl output/
echo "Build results:"
ls -lh output/*.whl
