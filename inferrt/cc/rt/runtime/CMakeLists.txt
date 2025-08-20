check_debug_log_out()

file(GLOB_RECURSE RUNTIME_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(runtime_obj STATIC ${RUNTIME_SRC_FILES})
target_link_libraries(runtime_obj tensor_obj ops_obj pass_obj kernel)