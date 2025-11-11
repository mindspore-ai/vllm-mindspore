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
PROJECT_DIR=$(pwd)
BUILD_DIR="${PROJECT_DIR}/build"
export LLVM_BUILD_DIR="${BUILD_DIR}/third_party/build/llvm"

if [[ $INC_BUILD != 1 && ($BUILD_OPT == 1 || $ENABLE_ASCEND == 1) ]]; then
    echo "LLVM_BUILD_DIR: ${LLVM_BUILD_DIR}"
fi

##################################################
# Install build dependencies
##################################################
python -c "import build" 2>/dev/null || {
    echo "Installing Python build package..."
    pip install build
}

# packaging>=24.2
python -c "
import packaging
from packaging.version import parse
assert parse(packaging.__version__) >= parse('24.2'), f'packaging {packaging.__version__} < 24.2'
" 2>/dev/null || pip install "packaging>=24.2"

##################################################
# Build mopt
##################################################
if [[ $BUILD_OPT == 1 ]]; then
    pushd mopt

    # Clean previous builds
    if [[ $INC_BUILD != 1 ]]; then
        rm -rf build dist
    fi

    export LLVM_DIR="${LLVM_BUILD_DIR}/lib/cmake/llvm"
    export MLIR_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir"
    echo "Building mopt with LLVM from: ${LLVM_BUILD_DIR}"
    echo "  LLVM_DIR: ${LLVM_DIR}"
    echo "  MLIR_DIR: ${MLIR_DIR}"
    echo "  ENABLE_TORCH_FRONT: ${ENABLE_TORCH_FRONT}"

    # Build the wheel
    python -m build --wheel --no-isolation

    popd
fi

# Generate ops code
if [[ $ENABLE_ASCEND == 1 ]]; then
    bash "scripts/gen_code.sh"
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
