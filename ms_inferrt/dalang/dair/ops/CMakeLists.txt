check_debug_log_out()

file(GLOB_RECURSE OPS_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(ops_obj STATIC ${OPS_SRC_FILES})