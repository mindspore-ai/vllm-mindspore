if("${DEBUG_LOG_OUT_ir}" STREQUAL "on")
add_definitions(-DDEBUG_LOG_OUT)
endif()

file(GLOB_RECURSE IR_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(ir_obj STATIC ${IR_SRC_FILES})