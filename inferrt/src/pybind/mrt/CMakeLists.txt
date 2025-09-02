check_debug_log_out()

# Add mrt_api pybind11 sub module
pybind11_add_module(_mrt_api NO_EXTRAS pybind11_api.cc)
target_link_libraries(_mrt_api PUBLIC inferrt)

set_target_properties(_mrt_api PROPERTIES
  INSTALL_RPATH "$ORIGIN:$ORIGIN/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(
  TARGETS _mrt_api
  LIBRARY DESTINATION .
  RUNTIME DESTINATION bin
)

# Add mrt_ir pybind11 sub module
pybind11_add_module(_mrt_ir NO_EXTRAS pybind11_ir.cc)
target_link_libraries(_mrt_ir PUBLIC inferrt)

# Add collective pybind11 sub module
if(ENABLE_ASCEND)
  if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
  else()
    set(ASCEND_PATH /usr/local/Ascend)
  endif()
  include_directories(${ASCEND_PATH}/latest/include/)
  include_directories(${ASCEND_PATH}/latest/lib64/)
  include_directories(${ASCEND_PATH}/latest/aarch64-linux/include/experiment)
  pybind11_add_module(_mrt_collective NO_EXTRAS pybind11_collective.cc)
  target_link_libraries(_mrt_collective PUBLIC inferrt hardware_ascend)
  set_target_properties(_mrt_collective PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
  install(
    TARGETS _mrt_collective
    LIBRARY DESTINATION .
    RUNTIME DESTINATION bin
  )
endif()

set_target_properties(_mrt_ir PROPERTIES
  INSTALL_RPATH "$ORIGIN:$ORIGIN/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(
  TARGETS _mrt_ir
  LIBRARY DESTINATION .
  RUNTIME DESTINATION bin
)
