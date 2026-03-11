check_debug_log_out()

# Add ms_inferrt_torch nanobind sub module
nanobind_add_module(_ms_inferrt_torch pybind_torch.cc)
target_link_libraries(_ms_inferrt_torch PUBLIC inferrt ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

if(ENABLE_ASCEND)
  if(DEFINED ENV{ASCEND_CUSTOM_PATH})
  set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
  else()
      set(ASCEND_PATH /usr/local/Ascend)
  endif()

  include_directories(${ASCEND_PATH}/latest/include/)

  add_compile_definitions(ENABLE_TORCH_NPU)
  include_directories(${TORCH_NPU_INCLUDE})
  target_link_directories(_ms_inferrt_torch PRIVATE ${TORCH_NPU_LIB_PATH})
  target_link_libraries(_ms_inferrt_torch PUBLIC torch_npu hardware_abstract)
endif()

set_target_properties(_ms_inferrt_torch PROPERTIES
  INSTALL_RPATH "$ORIGIN:$ORIGIN/lib:${TORCH_INSTALL_PREFIX}/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(
  TARGETS _ms_inferrt_torch
  LIBRARY DESTINATION .
  RUNTIME DESTINATION bin
)
