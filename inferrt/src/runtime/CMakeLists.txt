check_debug_log_out()
find_package(Threads REQUIRED)

file(GLOB_RECURSE RUNTIME_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(execution_engine SHARED ${RUNTIME_SRC_FILES})

target_link_libraries(execution_engine ops_obj ops_utils mrt_ir_obj kernel hardware_abstract pass_obj
                      Threads::Threads Python::Python nanobind-static)
target_include_directories(execution_engine PUBLIC ${NANOBIND_INCLUDE_DIR})

set_target_properties(execution_engine PROPERTIES
  INSTALL_RPATH "$ORIGIN:$ORIGIN/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(
  TARGETS execution_engine
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
