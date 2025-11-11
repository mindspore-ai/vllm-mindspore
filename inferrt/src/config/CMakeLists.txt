check_debug_log_out()

file(GLOB_RECURSE CONFIG_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(mrt_config SHARED ${CONFIG_SRC_FILES})
target_link_libraries(mrt_config PRIVATE mrt_common)

set_target_properties(mrt_config PROPERTIES
  INSTALL_RPATH "$ORIGIN:$ORIGIN/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(
  TARGETS mrt_config
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
