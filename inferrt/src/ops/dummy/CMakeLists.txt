check_debug_log_out()

file(GLOB_RECURSE DUMMY_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(kernel_dummy SHARED ${DUMMY_SRC_FILES})
target_link_libraries(kernel_dummy PRIVATE kernel)
