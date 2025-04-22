if("${DEBUG_LOG_OUT_ir}" STREQUAL "on")
add_definitions(-DDEBUG_LOG_OUT)
endif()

add_subdirectory(tensor)
add_subdirectory(ops)