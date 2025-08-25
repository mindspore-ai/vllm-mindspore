check_debug_log_out()

file(GLOB_RECURSE IR_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(ir_obj STATIC ${IR_SRC_FILES})