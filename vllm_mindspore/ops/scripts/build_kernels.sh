#!/bin/bash
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    cd ..
    pwd
)
cd $CURRENT_DIR

BUILD_TYPE="Debug"
INSTALL_PREFIX="${CURRENT_DIR}/build"
RUN_MODE="npu"

SHORT=r:,v:,i:,b:,p:,
LONG=run-mode:,soc-version:,install-path:,build-type:,install-prefix:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

while :; do
    case "$1" in
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    -i | --install-path)
        ASCEND_INSTALL_PATH="$2"
        shift 2
        ;;
    -b | --build-type)
        BUILD_TYPE="$2"
        shift 2
        ;;
    -p | --install-prefix)
        INSTALL_PREFIX="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR] Unexpected option: $1"
        break
        ;;
    esac
done

if [ -z "$SOC_VERSION" ]; then
    NPU_INFO=$(npu-smi info -t board -i 0 -c 0)

    CHIP_NAME=$(echo "$NPU_INFO" | grep 'Chip Name' | awk '{print $4}')

    CHIP_TYPE=$(echo "$NPU_INFO" | grep 'Chip Type' | awk '{print $4}')
    NPU_NAME=$(echo "$NPU_INFO" | grep 'NPU Name' | awk '{print $4}')

    if [ -n "$CHIP_TYPE" ]; then
        # ascend910b
        SOC_VERSION="${CHIP_TYPE}${CHIP_NAME}"
    elif [ -n "$NPU_NAME" ]; then
        # ascend910_93
        SOC_VERSION="${CHIP_NAME}_${NPU_NAME}"
    fi

    if [ -z "$SOC_VERSION" ]; then
        echo "[ERROR] Unable to determine SOC_VERSION. Please specify it explicitly using -v or --soc-version."
        exit 1
    fi
    echo "The soc version is: $SOC_VERSION"
fi

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}

source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash


set -e
rm -rf build out
mkdir -p build
cmake -B build \
    -DRUN_MODE=${RUN_MODE} \
    -DSOC_VERSION=${SOC_VERSION} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}
cmake --build build -j  --verbose
cmake --install build
