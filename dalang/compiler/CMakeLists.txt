file(GLOB_RECURSE COMPILER_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(compiler_obj STATIC ${COMPILER_SRC_FILES})
target_link_libraries(compiler_obj parser_obj)