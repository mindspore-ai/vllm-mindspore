check_debug_log_out()

# Link against torch libraries for _mrt_torch
execute_process(COMMAND python -c "import os; import torch; print(os.path.join(os.path.dirname(torch.__file__), 'share/cmake'))" OUTPUT_VARIABLE PYTORCH_CMAKE_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_PREFIX_PATH "${PYTORCH_CMAKE_PATH}")
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

# Add mrt_torch pybind11 sub module
pybind11_add_module(_mrt_torch NO_EXTRAS pybind11_torch.cc)
target_link_libraries(_mrt_torch PUBLIC inferrt ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
