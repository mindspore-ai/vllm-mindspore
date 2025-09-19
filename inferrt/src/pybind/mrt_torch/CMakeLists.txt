check_debug_log_out()

# Link against torch libraries for _mrt_torch
execute_process(COMMAND python -c "import os; import torch; print(os.path.join(os.path.dirname(torch.__file__), 'share/cmake'))" OUTPUT_VARIABLE PYTORCH_CMAKE_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_PREFIX_PATH "${PYTORCH_CMAKE_PATH}")
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

# Add mrt_torch pybind11 sub module
pybind11_add_module(_mrt_torch NO_EXTRAS pybind11_torch.cc)
target_link_libraries(_mrt_torch PUBLIC inferrt ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

if(ENABLE_ASCEND)
  add_compile_definitions(ENABLE_TORCH_NPU)
  execute_process(COMMAND python -c "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))" OUTPUT_VARIABLE TORCH_NPU_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
  message("TORCH_NPU_PATH: ${TORCH_NPU_PATH}")

  include_directories(${TORCH_NPU_PATH}/include/)
  target_link_directories(_mrt_torch PRIVATE ${TORCH_NPU_PATH}/lib/)
  target_link_libraries(_mrt_torch PUBLIC torch_npu)
endif()

set_target_properties(_mrt_torch PROPERTIES
  INSTALL_RPATH "$ORIGIN:$ORIGIN/lib:${TORCH_INSTALL_PREFIX}/lib"
  BUILD_WITH_INSTALL_RPATH TRUE  
)

install(
  TARGETS _mrt_torch
  LIBRARY DESTINATION .
  RUNTIME DESTINATION bin
)
