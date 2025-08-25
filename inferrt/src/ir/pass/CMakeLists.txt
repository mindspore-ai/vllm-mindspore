check_debug_log_out()

file(GLOB_RECURSE PASS_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(pass_obj STATIC ${PASS_SRC_FILES})
