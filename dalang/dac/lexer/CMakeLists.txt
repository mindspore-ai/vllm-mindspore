if("${DEBUG_LOG_OUT_lexer}" STREQUAL "on")
add_definitions(-DDEBUG_LOG_OUT)
endif()

file(GLOB_RECURSE LEXER_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(lexer_obj STATIC ${LEXER_SRC_FILES})