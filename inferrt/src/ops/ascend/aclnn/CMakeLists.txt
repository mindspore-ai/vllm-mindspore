check_debug_log_out()

add_subdirectory(utils)
file(GLOB OPS_ACLNN_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(ops_ascend_aclnn SHARED ${OPS_ACLNN_SRC_FILES})
target_link_libraries(ops_ascend_aclnn PRIVATE ops_ascend_aclnn_common ops_base_obj hardware_ascend kernel mrt::securec)

set_target_properties(ops_ascend_aclnn PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS ops_ascend_aclnn
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
