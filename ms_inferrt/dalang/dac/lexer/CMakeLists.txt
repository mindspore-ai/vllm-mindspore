check_debug_log_out()

file(GLOB_RECURSE LEXER_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(lexer_obj STATIC ${LEXER_SRC_FILES})