check_debug_log_out()

file(GLOB_RECURSE OPS_BASE_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(ops_base_obj STATIC ${OPS_BASE_SRC_FILES})
target_link_libraries(ops_base_obj PRIVATE mrt_ir_obj)
