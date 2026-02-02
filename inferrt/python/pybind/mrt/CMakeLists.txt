check_debug_log_out()

# Add mrt_api nanobind sub module
nanobind_add_module(_mrt_api pybind_api.cc)
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

# Add mrt_ir nanobind sub module
nanobind_add_module(_mrt_ir pybind_ir.cc)
target_link_libraries(_mrt_ir PUBLIC inferrt)

# Add collective nanobind sub module
nanobind_add_module(_mrt_collective pybind_collective.cc)
target_link_libraries(_mrt_collective PRIVATE inferrt)
set_target_properties(_mrt_collective PROPERTIES
  INSTALL_RPATH "$ORIGIN:$ORIGIN/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)
install(
  TARGETS _mrt_collective
  LIBRARY DESTINATION .
  RUNTIME DESTINATION bin
)

# Add mrt_config nanobind sub module
nanobind_add_module(_mrt_config pybind_config.cc)
target_link_libraries(_mrt_config PRIVATE inferrt)
set_target_properties(_mrt_config PROPERTIES
  INSTALL_RPATH "$ORIGIN:$ORIGIN/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)
install(
  TARGETS _mrt_config
  LIBRARY DESTINATION .
  RUNTIME DESTINATION bin
)

set_target_properties(_mrt_ir PROPERTIES
  INSTALL_RPATH "$ORIGIN:$ORIGIN/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(
  TARGETS _mrt_ir
  LIBRARY DESTINATION .
  RUNTIME DESTINATION bin
)
