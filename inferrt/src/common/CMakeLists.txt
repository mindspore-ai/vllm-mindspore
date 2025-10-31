check_debug_log_out()

file(GLOB_RECURSE COMMON_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(mrt_common SHARED ${COMMON_SRC_FILES})
target_link_libraries(mrt_common dl)
