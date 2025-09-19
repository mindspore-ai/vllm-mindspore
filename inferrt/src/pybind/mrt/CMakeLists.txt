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

set_target_properties(_mrt_ir PROPERTIES
  INSTALL_RPATH "$ORIGIN:$ORIGIN/lib"
  BUILD_WITH_INSTALL_RPATH TRUE  
)

install(
  TARGETS _mrt_ir
  LIBRARY DESTINATION .
  RUNTIME DESTINATION bin
)