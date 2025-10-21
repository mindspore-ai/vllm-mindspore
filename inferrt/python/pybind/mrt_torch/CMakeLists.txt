check_debug_log_out()

# Add mrt_torch pybind11 sub module
pybind11_add_module(_mrt_torch NO_EXTRAS pybind11_torch.cc)
target_link_libraries(_mrt_torch PUBLIC inferrt ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

if(ENABLE_ASCEND)
  add_compile_definitions(ENABLE_TORCH_NPU)
  include_directories(${TORCH_NPU_INCLUDE})
  target_link_directories(_mrt_torch PRIVATE ${TORCH_NPU_LIB_PATH})
  target_link_libraries(_mrt_torch PUBLIC torch_npu hardware_abstract)
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
