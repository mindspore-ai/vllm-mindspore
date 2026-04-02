check_debug_log_out()

set(CONFIG_SRC_FILES
  device/ascend/op_precision_conf.cc
)
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

add_library(mrt_aclgraph_config SHARED
  device/ascend/aclgraph_conf.cc
)
target_link_libraries(mrt_aclgraph_config PRIVATE execution_engine)

set_target_properties(mrt_aclgraph_config PROPERTIES
  INSTALL_RPATH "$ORIGIN:$ORIGIN/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(
  TARGETS mrt_aclgraph_config
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
