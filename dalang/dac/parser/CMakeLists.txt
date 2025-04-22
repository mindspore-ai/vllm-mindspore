if("${DEBUG_LOG_OUT_parser}" STREQUAL "on")
add_definitions(-DDEBUG_LOG_OUT)
endif()

file(GLOB_RECURSE PARSER_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(parser_obj STATIC ${PARSER_SRC_FILES})
target_link_libraries(parser_obj lexer_obj)