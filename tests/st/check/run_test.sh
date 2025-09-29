CURRENT_PATH=$(dirname $(dirname $(dirname $(dirname $(realpath "$0")))))
python $CURRENT_PATH/tests/st/check/check_backend.py

export MRT_ENABLE_PIPELINE="on"
python $CURRENT_PATH/tests/st/check/check_backend.py