check_debug_log_out()

# Add ms_inferrt_api nanobind sub module
nanobind_add_module(_ms_inferrt_api pybind_api.cc)
target_link_libraries(_ms_inferrt_api PUBLIC inferrt)

set_target_properties(_ms_inferrt_api PROPERTIES
  INSTALL_RPATH "$ORIGIN:$ORIGIN/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(
  TARGETS _ms_inferrt_api
  LIBRARY DESTINATION .
  RUNTIME DESTINATION bin
)

# Add ms_inferrt_ir nanobind sub module
nanobind_add_module(_ms_inferrt_ir pybind_ir.cc)
target_link_libraries(_ms_inferrt_ir PUBLIC inferrt)

# Add collective nanobind sub module
nanobind_add_module(_ms_inferrt_collective pybind_collective.cc)
target_link_libraries(_ms_inferrt_collective PRIVATE inferrt)
set_target_properties(_ms_inferrt_collective PROPERTIES
  INSTALL_RPATH "$ORIGIN:$ORIGIN/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)
install(
  TARGETS _ms_inferrt_collective
  LIBRARY DESTINATION .
  RUNTIME DESTINATION bin
)

# Add ms_inferrt_config nanobind sub module
nanobind_add_module(_ms_inferrt_config pybind_config.cc)
target_link_libraries(_ms_inferrt_config PRIVATE inferrt)
set_target_properties(_ms_inferrt_config PROPERTIES
  INSTALL_RPATH "$ORIGIN:$ORIGIN/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)
install(
  TARGETS _ms_inferrt_config
  LIBRARY DESTINATION .
  RUNTIME DESTINATION bin
)

set_target_properties(_ms_inferrt_ir PROPERTIES
  INSTALL_RPATH "$ORIGIN:$ORIGIN/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(
  TARGETS _ms_inferrt_ir
  LIBRARY DESTINATION .
  RUNTIME DESTINATION bin
)
