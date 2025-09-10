CURRENT_PATH=$(dirname $(dirname $(dirname $(dirname $(realpath "$0")))))
INFERRT_PATH=$CURRENT_PATH
BUILD_DIR=$CURRENT_PATH/build
export DART_KERNEL_LIB_PATH=$BUILD_DIR/inferrt/src/ops/cpu/aten/libkernel_aten.so
export DART_KERNEL_LIB_NAME=Aten
python $CURRENT_PATH/tests/st/check/check_backend.py