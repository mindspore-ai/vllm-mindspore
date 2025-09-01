check_debug_log_out()

file(GLOB_RECURSE OPS_ACLNN_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_library(ops_ascend_aclnn SHARED ${OPS_ACLNN_SRC_FILES})
target_link_libraries(ops_ascend_aclnn PRIVATE ops_base_obj hardware_ascend kernel mrt::securec)
