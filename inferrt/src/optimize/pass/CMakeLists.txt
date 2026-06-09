
file(GLOB_RECURSE PASS_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(pass_obj STATIC ${PASS_SRC_FILES})
target_link_libraries(pass_obj PRIVATE mrt_common)
