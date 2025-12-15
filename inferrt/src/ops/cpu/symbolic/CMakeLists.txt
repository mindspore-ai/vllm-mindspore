check_debug_log_out()

file(GLOB_RECURSE OPS_SYMBOLIC_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_library(ops_cpu_symbolic SHARED ${OPS_SYMBOLIC_SRC_FILES})
target_link_libraries(ops_cpu_symbolic PRIVATE ops_base_obj kernel)

set_target_properties(ops_cpu_symbolic PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS ops_cpu_symbolic
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
