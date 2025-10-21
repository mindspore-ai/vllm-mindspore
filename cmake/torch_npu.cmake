if(ENABLE_TORCH_FRONT OR ENABLE_ASCEND)
  execute_process(COMMAND python -c
  "import os; import torch; print(os.path.join(os.path.dirname(torch.__file__), 'share/cmake'))"
  OUTPUT_VARIABLE PYTORCH_CMAKE_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)

  set(CMAKE_PREFIX_PATH "${PYTORCH_CMAKE_PATH}")
  find_package(Torch REQUIRED)
  find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
endif()

if(ENABLE_ASCEND)
  execute_process(COMMAND python -c
  "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
  OUTPUT_VARIABLE TORCH_NPU_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
  message("TORCH_NPU_PATH: ${TORCH_NPU_PATH}")

  set(TORCH_NPU_INCLUDE ${TORCH_NPU_PATH}/include/)
  set(TORCH_NPU_LIB_PATH ${TORCH_NPU_PATH}/lib/)
endif()
