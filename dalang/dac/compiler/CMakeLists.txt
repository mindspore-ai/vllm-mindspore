if("${DEBUG_LOG_OUT_compiler}" STREQUAL "on")
add_definitions(-DDEBUG_LOG_OUT)
endif()

file(GLOB_RECURSE COMPILER_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(compiler_obj STATIC ${COMPILER_SRC_FILES})
target_link_libraries(compiler_obj parser_obj)